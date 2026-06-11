"""
特征工程 + 多模型训练（基于打卡时间数据）

三大分析视角：
① 离职员工：离职前一个月 vs 工作稳定期（刚入职中期）的打卡偏差
② 同部门对比：个人行为与部门平均水平的偏离度
③ 个人趋势：打卡行为的波动性、迟到/早退/加班频率的变化趋势

模型策略：Random Forest + XGBoost + Logistic Regression 集成
RF预选Top22特征缩减维度 + 三模型集成 + 3折CV + 样本权重。
特征砍掉37%（35→22）作为主要正则化手段，XGBoost保持适度内部正则化。
"""
import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
import re
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, 'attendance.db')
MODEL_DIR = os.path.join(BASE_DIR, 'models')


# =====================================================================
#  月度考勤聚合函数
# =====================================================================
def monthly_agg(emp_df):
    """计算单个员工各月度的考勤指标"""
    results = []
    for (yr, mo), grp in emp_df.groupby(['year', 'month']):
        n = len(grp)
        early_arr   = (grp['morning_status'] == '早到').sum()
        normal_morn = (grp['morning_status'] == '正常上班').sum()
        late_arr    = (grp['morning_status'] == '迟到').sum()
        normal_eve  = (grp['evening_status'] == '正常下班').sum()
        early_leave = (grp['evening_status'] == '早退').sum()
        overtime    = (grp['evening_status'] == '加班').sum()
        miss_morn   = (grp['morning_status'] == '上班缺卡').sum()
        miss_eve    = (grp['evening_status'] == '下班缺卡').sum()

        # 上班时间统计
        morn_times = grp['first_time_min'].dropna()
        avg_morn   = morn_times.mean() if len(morn_times) > 0 else np.nan
        std_morn   = morn_times.std() if len(morn_times) > 1 else 0

        # 下班时间统计
        eve_times = grp['last_time_min'].dropna()
        avg_eve   = eve_times.mean() if len(eve_times) > 0 else np.nan
        std_eve   = eve_times.std() if len(eve_times) > 1 else 0

        results.append({
            'year': yr, 'month': mo,
            'total_days': n,
            'early_arr_rate': early_arr / n,
            'normal_morn_rate': normal_morn / n,
            'late_arr_rate': late_arr / n,
            'normal_eve_rate': normal_eve / n,
            'early_leave_rate': early_leave / n,
            'overtime_rate': overtime / n,
            'miss_morn_rate': miss_morn / n,
            'miss_eve_rate': miss_eve / n,
            'avg_morning_min': avg_morn,
            'avg_evening_min': avg_eve,
            'std_morning': std_morn,
            'std_evening': std_eve,
        })
    return pd.DataFrame(results)


# =====================================================================
#  特征提取
# =====================================================================
def extract_features():
    """三大视角特征提取"""
    print("=" * 60)
    print("特征工程（三大视角）...")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    df_clock = pd.read_sql("SELECT * FROM clock_records ORDER BY date", conn)
    df_roster = pd.read_sql("SELECT * FROM employee_roster", conn)

    df_roster_subset = df_roster[['name', 'status', 'department', 'rank',
                                   'entry_date', 'resign_date', 'resign_reason',
                                   'gender']].copy()
    df_roster_subset['is_resigned'] = (df_roster_subset['status'] == '离职').astype(int)

    active_names = df_clock['name'].unique()
    df_roster_subset = df_roster_subset[df_roster_subset['name'].isin(active_names)]
    print(f"考勤员工: {len(active_names)}, 匹配花名册: {len(df_roster_subset)}")

    # === 部门平均考勤（视角②对照基准） ===
    dept_map = df_roster[['name', 'department']].drop_duplicates().set_index('name')['department'].to_dict()
    df_clock['department'] = df_clock['name'].map(dept_map).fillna('未知')

    # 计算部门各月度的平均考勤指标
    dept_monthly = df_clock.groupby(['department', 'year', 'month']).agg(
        dept_early_arr_rate=('morning_status', lambda x: (x == '早到').mean()),
        dept_late_arr_rate=('morning_status', lambda x: (x == '迟到').mean()),
        dept_normal_morn_rate=('morning_status', lambda x: (x == '正常上班').mean()),
        dept_early_leave_rate=('evening_status', lambda x: (x == '早退').mean()),
        dept_overtime_rate=('evening_status', lambda x: (x == '加班').mean()),
        dept_normal_eve_rate=('evening_status', lambda x: (x == '正常下班').mean()),
        dept_miss_rate=('evening_status', lambda x: (x == '下班缺卡').mean()),
    ).reset_index()

    # === 遍历每个员工提取特征 ===
    all_features = []

    for _, emp_row in df_roster_subset.iterrows():
        name = emp_row['name']
        emp_df = df_clock[df_clock['name'] == name].sort_values('date')

        if len(emp_df) < 10:  # 至少需要足够数据
            continue

        dept = dept_map.get(name, '未知')
        is_resigned = emp_row['is_resigned']
        resign_date = emp_row['resign_date']

        # 月度聚合
        monthly = monthly_agg(emp_df)
        if len(monthly) < 2:
            continue

        # ==================== ① 稳定期 vs 离职前特征 ====================
        n_recent = min(3, len(monthly) // 2)

        recent = monthly.tail(n_recent)
        early_period = monthly.head(len(monthly) - n_recent)
        if len(early_period) == 0:
            early_period = recent

        # 近期 vs 早期的偏差
        def diff_means(col):
            return recent[col].mean() - early_period[col].mean()

        # 行为恶化斜率（线性回归系数 × 100，标准化量纲）
        def calc_slope(values):
            if len(values) < 2:
                return 0.0
            x = np.arange(len(values))
            mask = ~np.isnan(values)
            if mask.sum() < 2:
                return 0.0
            return np.polyfit(x[mask], values[mask], 1)[0]

        # 上下班行为协同模式（从日数据计算）
        n_days = len(emp_df)
        if n_days > 0:
            pos_pattern = ((emp_df['morning_status'] == '早到') &
                           (emp_df['evening_status'] == '加班')).sum() / n_days
            neg_pattern = ((emp_df['morning_status'] == '迟到') &
                           (emp_df['evening_status'] == '早退')).sum() / n_days
            damage_ctrl  = ((emp_df['morning_status'] == '迟到') &
                           (emp_df['evening_status'] == '加班')).sum() / n_days
            full_absent  = ((emp_df['morning_status'] == '上班缺卡') &
                           (emp_df['evening_status'] == '下班缺卡')).sum() / n_days
        else:
            pos_pattern = neg_pattern = damage_ctrl = full_absent = 0.0

        feats = {
            'name': name,
            'is_resigned': is_resigned,
            'department': dept,
            'total_months': len(monthly),

            # ── 近期均值（近3个月） ──
            'recent_early_arr': recent['early_arr_rate'].mean(),
            'recent_late_arr': recent['late_arr_rate'].mean(),
            'recent_early_leave': recent['early_leave_rate'].mean(),
            'recent_overtime': recent['overtime_rate'].mean(),
            'recent_miss_eve': recent['miss_eve_rate'].mean(),
            'recent_avg_morning': recent['avg_morning_min'].mean(),
            'recent_avg_evening': recent['avg_evening_min'].mean(),

            # ── 最后一个月独立指标（不混合前两个月） ──
            'last_month_late_arr': monthly.iloc[-1]['late_arr_rate'],
            'last_month_overtime': monthly.iloc[-1]['overtime_rate'],
            'last_month_miss_eve': monthly.iloc[-1]['miss_eve_rate'],

            # ── 趋势（近期 - 早期） ──
            'trend_early_arr': diff_means('early_arr_rate'),
            'trend_late_arr': diff_means('late_arr_rate'),
            'trend_early_leave': diff_means('early_leave_rate'),
            'trend_overtime': diff_means('overtime_rate'),
            'trend_miss_eve': diff_means('miss_eve_rate'),

            # ── 行为恶化斜率（所有月份的线性趋势） ──
            'slope_late_arr': calc_slope(monthly['late_arr_rate'].values),
            'slope_overtime': calc_slope(monthly['overtime_rate'].values),
            'slope_miss_eve': calc_slope(monthly['miss_eve_rate'].values),

            # ── 整体波动（标准差+变异系数） ──
            'std_early_arr': monthly['early_arr_rate'].std(),
            'std_late_arr': monthly['late_arr_rate'].std(),
            'std_early_leave': monthly['early_leave_rate'].std(),
            'std_overtime': monthly['overtime_rate'].std(),
            'cv_late_arr': (monthly['late_arr_rate'].std() /
                            max(monthly['late_arr_rate'].mean(), 0.001)),
            'cv_overtime': (monthly['overtime_rate'].std() /
                            max(monthly['overtime_rate'].mean(), 0.001)),

            # ── 上下班行为协同模式 ──
            'pos_pattern_rate': pos_pattern,
            'neg_pattern_rate': neg_pattern,
            'damage_control_rate': damage_ctrl,
            'full_absent_rate': full_absent,
        }

        # ==================== ② 同部门偏差特征 ====================
        emp_dept_monthly = dept_monthly[
            (dept_monthly['department'] == dept)
        ]
        # 只取该员工有数据的月份
        emp_ym = set(zip(monthly['year'], monthly['month']))
        dept_ref = emp_dept_monthly[
            emp_dept_monthly.apply(lambda r: (r['year'], r['month']) in emp_ym, axis=1)
        ]

        if len(dept_ref) > 0:
            dept_early_arr_avg = dept_ref['dept_early_arr_rate'].mean()
            dept_late_avg = dept_ref['dept_late_arr_rate'].mean()
            dept_early_leave_avg = dept_ref['dept_early_leave_rate'].mean()
            dept_overtime_avg = dept_ref['dept_overtime_rate'].mean()
            dept_miss_avg = dept_ref['dept_miss_rate'].mean()
        else:
            dept_early_arr_avg = 0
            dept_late_avg = 0
            dept_early_leave_avg = 0
            dept_overtime_avg = 0
            dept_miss_avg = 0

        feats['dev_early_arr'] = feats['recent_early_arr'] - dept_early_arr_avg
        feats['dev_late_arr'] = feats['recent_late_arr'] - dept_late_avg
        feats['dev_early_leave'] = feats['recent_early_leave'] - dept_early_leave_avg
        feats['dev_overtime'] = feats['recent_overtime'] - dept_overtime_avg
        feats['dev_miss'] = feats['recent_miss_eve'] - dept_miss_avg

        all_features.append(feats)

    df_features = pd.DataFrame(all_features).fillna(0)
    print(f"\n员工特征: {len(df_features)} 人")
    print(f"  离职: {df_features['is_resigned'].sum():.0f}")
    print(f"  在职: {(1 - df_features['is_resigned']).sum():.0f}")
    print(f"  特征维度: {len([c for c in df_features.columns if c not in ('name','is_resigned','department')])}")

    conn.close()
    return df_features


# =====================================================================
#  模型训练（三模型集成）
# =====================================================================
def train_models(df_features):
    """训练 RF + XGB + LR 三模型"""
    print("\n" + "=" * 60)
    print("模型训练（三模型集成）...")
    print("=" * 60)

    cat_cols = ['department']
    df = df_features.copy()

    # 编码分类特征
    label_encoders = {}
    for col in cat_cols:
        df[col] = df[col].fillna('未知').astype(str)
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le

    # 特征列
    exclude = {'name', 'is_resigned', 'department'}
    feature_cols = [c for c in df.columns if c not in exclude]
    # 确保 department_encoded 只出现一次
    feature_cols = list(dict.fromkeys(feature_cols))

    X = df[feature_cols].fillna(0)
    y = df['is_resigned'].values

    print(f"特征: {X.shape}")
    print(f"  离职: {y.sum():.0f}, 在职: {(1-y).sum():.0f}")
    print(f"特征列表 ({len(feature_cols)}): {feature_cols}")

    # =====================================================================
    #  10 次重复随机划分评估（完整流水线：选特征→缩放→训练→测试）
    # =====================================================================
    REPEATS = 10
    seeds = [0, 10, 42, 100, 200, 500, 999, 2024, 12345, 88888]
    all_metrics = {m: {'test_auc': [], 'test_f1': []} for m in ['rf', 'xgb', 'lr']}

    print("\n" + "=" * 70)
    print(f"  重复 {REPEATS} 次随机划分评估")
    print("=" * 70)

    for rep, seed in enumerate(seeds):
        # ── 分层划分（保持类别比例） ──
        X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y
        )

        # ── RF 特征选择（在训练集上，Top 22） ──
        selector = RandomForestClassifier(
            n_estimators=200, max_depth=3, random_state=seed, n_jobs=-1,
            class_weight='balanced',
        )
        selector.fit(X_tr_raw, y_tr)
        importances = pd.Series(selector.feature_importances_, index=feature_cols)
        top_k = min(22, len(feature_cols))
        sel = importances.nlargest(top_k).index.tolist()

        # ── 提取特征子集 + 缩放 ──
        scaler = StandardScaler()
        X_tr_sel = scaler.fit_transform(X_tr_raw[sel])
        X_te_sel = scaler.transform(X_te_raw[sel])

        # ── 样本权重 ──
        sw = compute_sample_weight('balanced', y_tr)

        # ── 三折 CV（用于 XGB 找最优阈值） ──
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        cv_y, cv_prob = [], []

        # ── 1. RF ──
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=3, min_samples_leaf=5,
            min_samples_split=8, max_features='log2',
            class_weight='balanced', random_state=seed, n_jobs=-1,
        )
        rf.fit(X_tr_sel, y_tr)
        pp_rf = rf.predict_proba(X_te_sel)[:, 1]
        yp_rf = (pp_rf >= 0.5).astype(int)
        all_metrics['rf']['test_auc'].append(roc_auc_score(y_te, pp_rf))
        all_metrics['rf']['test_f1'].append(f1_score(y_te, yp_rf))

        # ── 2. XGBoost ──
        xgb_params = dict(
            n_estimators=80, max_depth=3, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, reg_alpha=0.2, reg_lambda=0.5,
            gamma=0.05,
            random_state=seed, n_jobs=-1, verbosity=0, eval_metric='logloss',
        )
        cv_y_fold, cv_prob_fold = [], []
        for train_idx, val_idx in inner_cv.split(X_tr_sel, y_tr):
            X_f, y_f = X_tr_sel[train_idx], y_tr[train_idx]
            X_v, y_v = X_tr_sel[val_idx], y_tr[val_idx]
            w_f = compute_sample_weight('balanced', y_f)
            xgb_f = xgb.XGBClassifier(**xgb_params)
            xgb_f.fit(X_f, y_f, sample_weight=w_f)
            pp_v = xgb_f.predict_proba(X_v)[:, 1]
            cv_y_fold.extend(y_v)
            cv_prob_fold.extend(pp_v)
        # 最优阈值
        best_th, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.96, 0.01):
            yp = (np.array(cv_prob_fold) >= t).astype(int)
            f1v = f1_score(cv_y_fold, yp)
            if f1v > best_f1:
                best_f1 = f1v
                best_th = t
        xgb_m = xgb.XGBClassifier(**xgb_params)
        xgb_m.fit(X_tr_sel, y_tr, sample_weight=sw)
        pp_xgb = xgb_m.predict_proba(X_te_sel)[:, 1]
        yp_xgb = (pp_xgb >= best_th).astype(int)
        all_metrics['xgb']['test_auc'].append(roc_auc_score(y_te, pp_xgb))
        all_metrics['xgb']['test_f1'].append(f1_score(y_te, yp_xgb))

        # ── 3. LR（优化：共享22→Top12 + GridSearch C/penalty + 阈值调优） ──
        # 从已选 22 特征中取 Top 12
        fc_list = list(feature_cols)
        sel_imp = [selector.feature_importances_[fc_list.index(f)] for f in sel]
        imp_lr = pd.Series(sel_imp, index=sel)
        top12_names = imp_lr.nlargest(12).index.tolist()
        top12_loci = [sel.index(f) for f in top12_names]
        X_tr_lr_loop = X_tr_sel[:, top12_loci]
        X_te_lr_loop = X_te_sel[:, top12_loci]

        gs_loop = GridSearchCV(
            LogisticRegression(class_weight='balanced', solver='liblinear',
                               random_state=seed, max_iter=1000),
            param_grid={'C': [0.01, 0.05, 0.1, 0.5, 1.0],
                        'penalty': ['l1', 'l2']},
            cv=StratifiedKFold(3, shuffle=True, random_state=seed),
            scoring='roc_auc',
        )
        gs_loop.fit(X_tr_lr_loop, y_tr)
        lr = gs_loop.best_estimator_

        # CV 上找最优阈值
        lr_cv_p, lr_cv_y = [], []
        for ti, vi in StratifiedKFold(3, shuffle=True, random_state=seed).split(X_tr_lr_loop, y_tr):
            lr_f = LogisticRegression(C=lr.C, penalty=lr.penalty, solver='liblinear',
                                      class_weight='balanced', random_state=seed, max_iter=1000)
            lr_f.fit(X_tr_lr_loop[ti], y_tr[ti])
            lr_cv_p.extend(lr_f.predict_proba(X_tr_lr_loop[vi])[:, 1])
            lr_cv_y.extend(y_tr[vi])
        lr_best_th = 0.5
        best_f1_lr = 0.0
        for t in np.arange(0.05, 0.96, 0.01):
            yp = (np.array(lr_cv_p) >= t).astype(int)
            f1v = f1_score(lr_cv_y, yp)
            if f1v > best_f1_lr:
                best_f1_lr = f1v
                lr_best_th = t
        lr.fit(X_tr_lr_loop, y_tr)
        pp_lr = lr.predict_proba(X_te_lr_loop)[:, 1]
        yp_lr = (pp_lr >= lr_best_th).astype(int)
        all_metrics['lr']['test_auc'].append(roc_auc_score(y_te, pp_lr))
        all_metrics['lr']['test_f1'].append(f1_score(y_te, yp_lr))

    # ── 打印汇总 ──
    print("\n" + "=" * 70)
    print(f"  10 次重复划分测试集评估汇总")
    print("=" * 70)
    print(f"{'模型':>10} | {'测试 AUC 均值':>12} {'标准差':>8} | {'测试 F1 均值':>12} {'标准差':>8}")
    print("-" * 70)
    for label, key in [('RF', 'rf'), ('XGBoost', 'xgb'), ('LR', 'lr')]:
        aucs = all_metrics[key]['test_auc']
        f1s = all_metrics[key]['test_f1']
        print(f"{label:>10} | {np.mean(aucs):>8.4f}  ±{np.std(aucs):<6.4f} | {np.mean(f1s):>8.4f}  ±{np.std(f1s):<6.4f}")

    # =====================================================================
    #  最终训练（seed=42）→ 保存模型
    # =====================================================================
    print("\n" + "=" * 60)
    print("最终模型训练（用于部署）...")
    print("=" * 60)

    FINAL_SEED = 42
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=FINAL_SEED, stratify=y
    )
    print(f"\n训练: {X_tr_raw.shape[0]}, 测试: {X_te_raw.shape[0]}")

    # 特征选择
    original_feature_cols = feature_cols.copy()
    selector = RandomForestClassifier(
        n_estimators=200, max_depth=3, random_state=FINAL_SEED, n_jobs=-1,
        class_weight='balanced',
    )
    selector.fit(X_tr_raw, y_tr)
    importances = pd.Series(selector.feature_importances_, index=feature_cols)
    top_k = min(22, len(feature_cols))
    sel_feats = importances.nlargest(top_k).index.tolist()
    print(f"\n最终选择 Top {top_k} / {len(feature_cols)}:")
    for f in sel_feats:
        print(f"  {f}: {importances[f]:.4f}")
    feature_cols = sel_feats

    # 缩放
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw[sel_feats])
    X_te = scaler.transform(X_te_raw[sel_feats])
    print(f"训练 {X_tr.shape[0]}, 测试 {X_te.shape[0]}, 特征 {X_tr.shape[1]}")

    sw = compute_sample_weight('balanced', y_tr)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=FINAL_SEED)

    # ── RF ──
    print("\n--- RF ---")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=3, min_samples_leaf=5,
        min_samples_split=8, max_features='log2',
        class_weight='balanced', random_state=FINAL_SEED, n_jobs=-1,
    )
    cv_scores_rf = cross_val_score(rf, X_tr, y_tr, cv=cv, scoring='roc_auc')
    print(f"  CV AUC: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")
    rf.fit(X_tr, y_tr)
    yp_rf = rf.predict(X_te)
    pp_rf = rf.predict_proba(X_te)[:, 1]
    print(f"  测试 F1={f1_score(y_te, yp_rf):.4f}, AUC={roc_auc_score(y_te, pp_rf):.4f}")

    # ── XGB ──
    print("\n--- XGB ---")
    xgb_params = dict(
        n_estimators=80, max_depth=3, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_alpha=0.2, reg_lambda=0.5,
        gamma=0.05,
        random_state=FINAL_SEED, n_jobs=-1, verbosity=0, eval_metric='logloss',
    )
    cv_scores_xgb = []
    all_val_y, all_val_prob = [], []
    for train_idx, val_idx in cv.split(X_tr, y_tr):
        X_f, y_f = X_tr[train_idx], y_tr[train_idx]
        X_v, y_v = X_tr[val_idx], y_tr[val_idx]
        w_f = compute_sample_weight('balanced', y_f)
        xgb_f = xgb.XGBClassifier(**xgb_params)
        xgb_f.fit(X_f, y_f, sample_weight=w_f)
        pp_v = xgb_f.predict_proba(X_v)[:, 1]
        cv_scores_xgb.append(roc_auc_score(y_v, pp_v))
        all_val_y.extend(y_v)
        all_val_prob.extend(pp_v)
    print(f"  CV AUC: {np.mean(cv_scores_xgb):.4f} (+/- {np.std(cv_scores_xgb):.4f})")
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.01):
        yp = (np.array(all_val_prob) >= t).astype(int)
        f1v = f1_score(all_val_y, yp)
        if f1v > best_f1:
            best_f1 = f1v
            best_thresh = t
    print(f"  最优阈值: {best_thresh:.2f} (CV F1={best_f1:.4f})")
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_tr, y_tr, sample_weight=sw)
    pp_xgb = xgb_model.predict_proba(X_te)[:, 1]
    yp_xgb_opt = (pp_xgb >= best_thresh).astype(int)
    yp_xgb_default = (pp_xgb >= 0.5).astype(int)
    f1_opt = f1_score(y_te, yp_xgb_opt)
    print(f"  测试 F1(阈值0.5)={f1_score(y_te, yp_xgb_default):.4f}, F1(阈值{best_thresh:.2f})={f1_opt:.4f}, AUC={roc_auc_score(y_te, pp_xgb):.4f}")

    # ── LR（完整优化流水线，使用共享 22 特征） ──
    print("\n--- LR ---")

    # Step 1: 从 22 共享特征中选 Top 12（进一步精简）
    fc_list = list(feature_cols)
    # selector 有 35 个 importance，从原始 35 特征列表中索引
    sel_imp = [selector.feature_importances_[list(original_feature_cols).index(f)] for f in feature_cols]
    imp_lr = pd.Series(sel_imp, index=feature_cols)
    top12_names = imp_lr.nlargest(12).index.tolist()
    top12_idx = [list(feature_cols).index(f) for f in top12_names]
    X_tr_lr = X_tr[:, top12_idx]
    X_te_lr = X_te[:, top12_idx]
    print(f"  LR 特征 (Top12/22): {top12_names}")

    # Step 2: GridSearch C + penalty（3折CV）
    gs = GridSearchCV(
        LogisticRegression(class_weight='balanced', solver='liblinear',
                           random_state=FINAL_SEED, max_iter=1000),
        param_grid={'C': [0.01, 0.05, 0.1, 0.5, 1.0],
                    'penalty': ['l1', 'l2']},
        cv=StratifiedKFold(3, shuffle=True, random_state=FINAL_SEED),
        scoring='roc_auc',
        return_train_score=False,
    )
    gs.fit(X_tr_lr, y_tr)
    best_c = gs.best_params_['C']
    best_penalty = gs.best_params_['penalty']
    lr_cv_auc = gs.best_score_
    # 打印所有参数组合的 CV 分数
    print(f"  GridSearch 最佳: C={best_c}, penalty={best_penalty}, CV AUC={lr_cv_auc:.4f}")
    print(f"  所有参数 CV AUC:")
    for i, params in enumerate(gs.cv_results_['params']):
        means = gs.cv_results_['mean_test_score'][i]
        stds = gs.cv_results_['std_test_score'][i]
        print(f"    C={params['C']:>4}, penalty={params['penalty']:>2}: {means:.4f} ± {stds:.4f}")
    # 打印最佳模型的每折分数
    best_idx = gs.best_index_
    fold_scores = [gs.cv_results_[f'split{fold}_test_score'][best_idx] for fold in range(3)]
    print(f"  最佳参数 3 折 AUC: {[f'{s:.4f}' for s in fold_scores]}, 均值={lr_cv_auc:.4f}")

    # Step 3: 如果 CV AUC < 0.85，尝试多项式交互特征 + L1
    use_poly = False
    if lr_cv_auc < 0.85:
        print("  CV AUC < 0.85，尝试多项式交互特征...")
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_tr_poly = poly.fit_transform(X_tr_lr)
        X_te_poly = poly.transform(X_te_lr)
        scaler_poly = StandardScaler()
        X_tr_poly_s = scaler_poly.fit_transform(X_tr_poly)
        X_te_poly_s = scaler_poly.transform(X_te_poly)

        gs_poly = GridSearchCV(
            LogisticRegression(class_weight='balanced', solver='liblinear',
                               random_state=FINAL_SEED, max_iter=1000, penalty='l1'),
            param_grid={'C': [0.01, 0.05, 0.1, 0.5, 1.0]},
            cv=StratifiedKFold(3, shuffle=True, random_state=FINAL_SEED),
            scoring='roc_auc',
        )
        gs_poly.fit(X_tr_poly_s, y_tr)
        poly_auc = gs_poly.best_score_
        print(f"  多项式 CV AUC: {poly_auc:.4f} (C={gs_poly.best_params_['C']})")
        if poly_auc > lr_cv_auc:
            print(f"  → 改用多项式特征 (CV AUC 提升 {poly_auc - lr_cv_auc:.4f})")
            use_poly = True
            best_c = gs_poly.best_params_['C']
            best_penalty = 'l1'
            X_tr_lr, X_te_lr = X_tr_poly_s, X_te_poly_s
            lr_cv_auc = poly_auc

    # Step 4: 阈值调优优化 F1
    lr_cv_preds, lr_cv_y = [], []
    for tr_idx, va_idx in StratifiedKFold(3, shuffle=True, random_state=FINAL_SEED).split(X_tr_lr, y_tr):
        lr_f = LogisticRegression(C=best_c, penalty=best_penalty, solver='liblinear',
                                  class_weight='balanced', random_state=FINAL_SEED, max_iter=1000)
        lr_f.fit(X_tr_lr[tr_idx], y_tr[tr_idx])
        lr_cv_preds.extend(lr_f.predict_proba(X_tr_lr[va_idx])[:, 1])
        lr_cv_y.extend(y_tr[va_idx])
    lr_opt_th, lr_opt_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.01):
        yp = (np.array(lr_cv_preds) >= t).astype(int)
        f1v = f1_score(lr_cv_y, yp)
        if f1v > lr_opt_f1:
            lr_opt_f1 = f1v
            lr_opt_th = t
    print(f"  最优阈值: {lr_opt_th:.2f} (CV F1={lr_opt_f1:.4f})")

    # 最终训练 + 测试
    lr = LogisticRegression(C=best_c, penalty=best_penalty, solver='liblinear',
                            class_weight='balanced', random_state=FINAL_SEED, max_iter=1000)
    lr.fit(X_tr_lr, y_tr)
    pp_lr = lr.predict_proba(X_te_lr)[:, 1]
    yp_lr_opt = (pp_lr >= lr_opt_th).astype(int)
    yp_lr_default = (pp_lr >= 0.5).astype(int)
    print(f"  测试 F1(阈值0.5)={f1_score(y_te, yp_lr_default):.4f}, "
          f"F1(阈值{lr_opt_th:.2f})={f1_score(y_te, yp_lr_opt):.4f}, "
          f"AUC={roc_auc_score(y_te, pp_lr):.4f}")

    cv_scores_lr = [lr_cv_auc] * 3
    lr_feat_names = top12_names  # 供 predict_employee index

    # ── 特征重要性 ──
    rf_imp = pd.DataFrame({'feature': feature_cols, 'importance': rf.feature_importances_})
    print(f"\nRF Top 8:")
    for _, r in rf_imp.sort_values('importance', ascending=False).head(8).iterrows():
        print(f"  {r['feature']}: {r['importance']:.4f}")

    # ========== 保存 ==========
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_info = {
        'rf_model': rf, 'xgb_model': xgb_model, 'lr_model': lr,
        'scaler': scaler, 'label_encoders': label_encoders,
        'feature_cols': feature_cols, 'cat_cols': cat_cols,
        'rf_metrics':  {'f1': f1_score(y_te, yp_rf), 'auc': roc_auc_score(y_te, pp_rf),
                        'cv_auc': cv_scores_rf.mean(), 'cv_auc_std': cv_scores_rf.std()},
        'xgb_metrics': {'f1': f1_opt, 'auc': roc_auc_score(y_te, pp_xgb),
                        'cv_auc': np.mean(cv_scores_xgb), 'cv_auc_std': np.std(cv_scores_xgb),
                        'opt_threshold': best_thresh},
        'lr_metrics':  {'f1': f1_score(y_te, yp_lr_opt), 'auc': roc_auc_score(y_te, pp_lr),
                        'cv_auc': lr_cv_auc, 'cv_auc_std': 0,
                        'opt_threshold': lr_opt_th},
        'ensemble_weights': {
            'rf':  cv_scores_rf.mean() if not np.isnan(cv_scores_rf.mean()) else roc_auc_score(y_te, pp_rf),
            'xgb': np.mean(cv_scores_xgb) if not np.isnan(np.mean(cv_scores_xgb)) else roc_auc_score(y_te, pp_xgb),
            'lr':  lr_cv_auc if not np.isnan(lr_cv_auc) else roc_auc_score(y_te, pp_lr),
        },
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': len(X), 'n_features': X_tr.shape[1],
        'n_resigned': int(y.sum()),
        'n_active': len(X) - int(y.sum()),
        'n_test': len(y_te),
        'test_ratio': 0.25,
        'repeat_metrics': all_metrics,
        'lr_feat_names': lr_feat_names,
    }
    path = os.path.join(MODEL_DIR, 'attendance_models.pkl')
    joblib.dump(model_info, path)
    print(f"\n模型已保存: {path}")
    return model_info, df_features


def predict_employee(monthly_data_dict, model_info):
    """
    预测单个员工的离职风险
    monthly_data_dict: 包含该员工各月的考勤指标（来自 monthly_agg）
    model_info: 训练好的模型字典
    """
    rf = model_info['rf_model']
    xgb_m = model_info['xgb_model']
    lr = model_info['lr_model']
    scaler = model_info['scaler']
    fc = model_info['feature_cols']  # 模型实际使用的特征列表

    monthly = pd.DataFrame(monthly_data_dict)
    if len(monthly) < 2:
        return {'rf_prob': 0.5, 'xgb_prob': 0.5, 'lr_prob': 0.5, 'ensemble_prob': 0.5}

    n_recent = min(3, len(monthly) // 2)
    recent = monthly.tail(n_recent)
    early_ = monthly.head(len(monthly) - n_recent)
    if len(early_) == 0:
        early_ = recent

    def diff_means(col):
        return recent[col].mean() - early_[col].mean()

    def calc_slope(values):
        vals = values.dropna().values
        if len(vals) < 2:
            return 0.0
        x = np.arange(len(vals))
        return float(np.polyfit(x, vals, 1)[0])

    # 计算所有可能用到的值，按 fc 提取
    feats = {}
    for col in fc:
        feats[col] = 0.0

    # ── 识别特征名称并填充已知值 ──
    # 对于每个在 fc 中的特征，尝试计算
    for col in fc:
        if col == 'total_months':
            feats[col] = len(monthly)
        elif col == 'department_encoded':
            feats[col] = 0.0
        elif col.startswith('recent_'):
            # recent_early_arr, recent_late_arr, etc.
            key = col.replace('recent_', '')
            # 映射到 monthly 中的列名
            col_map = {
                'early_arr': 'early_arr_rate',
                'late_arr': 'late_arr_rate',
                'early_leave': 'early_leave_rate',
                'overtime': 'overtime_rate',
                'miss_eve': 'miss_eve_rate',
                'avg_morning': 'avg_morning_min',
                'avg_evening': 'avg_evening_min',
            }
            if key in col_map:
                feats[col] = recent[col_map[key]].mean()
        elif col.startswith('last_month_'):
            key = col.replace('last_month_', '')
            col_map = {
                'late_arr': 'late_arr_rate',
                'overtime': 'overtime_rate',
                'miss_eve': 'miss_eve_rate',
            }
            if key in col_map:
                feats[col] = monthly.iloc[-1][col_map[key]]
        elif col.startswith('trend_'):
            key = col.replace('trend_', '')
            col_map = {
                'early_arr': 'early_arr_rate',
                'late_arr': 'late_arr_rate',
                'early_leave': 'early_leave_rate',
                'overtime': 'overtime_rate',
                'miss_eve': 'miss_eve_rate',
            }
            if key in col_map:
                feats[col] = diff_means(col_map[key])
        elif col.startswith('slope_'):
            key = col.replace('slope_', '')
            col_map = {
                'late_arr': 'late_arr_rate',
                'overtime': 'overtime_rate',
                'miss_eve': 'miss_eve_rate',
            }
            if key in col_map:
                feats[col] = calc_slope(monthly[col_map[key]])
        elif col.startswith('cv_'):
            key = col.replace('cv_', '')
            col_map = {
                'late_arr': 'late_arr_rate',
                'overtime': 'overtime_rate',
            }
            if key in col_map:
                m = monthly[col_map[key]]
                feats[col] = m.std() / max(m.mean(), 0.001)
        elif col.startswith('std_'):
            key = col.replace('std_', '')
            col_map = {
                'early_arr': 'early_arr_rate',
                'late_arr': 'late_arr_rate',
                'early_leave': 'early_leave_rate',
                'overtime': 'overtime_rate',
            }
            if key in col_map:
                feats[col] = monthly[col_map[key]].std()
        elif col in ('pos_pattern_rate', 'neg_pattern_rate', 'damage_control_rate', 'full_absent_rate'):
            # 这部分需要 day-level 数据，predict 场景无此数据，默认 0
            feats[col] = 0.0
        elif col.startswith('dev_'):
            # 部门偏差需要部门均值，predict 场景无此对照，默认 0
            feats[col] = 0.0

    X_pred = pd.DataFrame([feats])[fc].fillna(0)
    X_s = scaler.transform(X_pred)

    p_rf = float(rf.predict_proba(X_s)[0, 1])
    p_xgb = float(xgb_m.predict_proba(X_s)[0, 1])

    # LR 使用独立特征子集（仅 Top12）
    lr_feats = model_info.get('lr_feat_names', fc)
    lr_idx = [list(fc).index(f) for f in lr_feats if f in fc]
    if len(lr_idx) == lr.coef_.shape[1]:
        X_s_lr = X_s[:, lr_idx]
    else:
        X_s_lr = X_s  # fallback
    p_lr = float(lr.predict_proba(X_s_lr)[0, 1])

    # 按 CV AUC 加权平均（兼容旧模型无 ensemble_weights / NaN 权重的情况）
    w = model_info.get('ensemble_weights', None)
    if w and not any(np.isnan(v) for v in [w['rf'], w['xgb'], w['lr']]):
        total = w['rf'] + w['xgb'] + w['lr']
        p_ens = (p_rf * w['rf'] + p_xgb * w['xgb'] + p_lr * w['lr']) / total
    else:
        p_ens = (p_rf + p_xgb + p_lr) / 3

    return {
        'rf_prob': p_rf, 'xgb_prob': p_xgb, 'lr_prob': p_lr,
        'ensemble_prob': p_ens,
    }


if __name__ == '__main__':
    df = extract_features()
    mi, _ = train_models(df)
    print("\n完成！")
