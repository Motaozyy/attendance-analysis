"""
打卡时间解析器：从Excel提取上下班打卡时间，按规则分类考勤状态
废弃旧的考勤状态关键词数据，仅使用打卡时间模式数据
"""
import pandas as pd
import numpy as np
import re
import sqlite3
import xlrd
import os
from datetime import datetime, timedelta, date as date_type
from chinese_calendar import is_workday

DB_PATH = os.path.join(os.path.dirname(__file__), 'attendance.db')
DATA_FILE = os.path.join(os.path.dirname(__file__), '202401-202605考勤 - 副本 - 副本(1).xls')

# 规则阈值（分钟）
MORNING_NORMAL_START = 8 * 60 + 30   # 08:30
MORNING_NORMAL_END   = 9 * 60 + 0    # 09:00
EVENING_NORMAL_START = 17 * 60 + 30  # 17:30
EVENING_NORMAL_END   = 18 * 60 + 0   # 18:00


def time_str_to_minutes(t_str):
    """转换时间字符串 'HH:MM' → 分钟数"""
    if not t_str or not isinstance(t_str, str):
        return None
    # 清理：移除空格和外勤标记
    t_str = t_str.strip().replace('外勤', '').replace(' ', '')
    m = re.match(r'(\d{1,2}):(\d{2})', t_str)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return None


NOON = 12 * 60  # 12:00 界线


def parse_cell_times_split(cell_val):
    """
    解析单元格打卡时间，按 12:00 分界区分上/下班。
    返回 (morning_times_list, evening_times_list)
    - 12:00 前（不含12:00）→ 上班打卡
    - 12:00（含）后 → 下班打卡
    - 空单元格 → ([], [])
    """
    if not cell_val or not isinstance(cell_val, str) or not cell_val.strip():
        return [], []  # 空单元格

    parts = cell_val.replace('\n', ' ').split()
    morning_times = []
    evening_times = []

    for p in parts:
        p = p.strip()
        t = time_str_to_minutes(p)
        if t is not None:
            if t < NOON:
                morning_times.append(t)
            else:
                evening_times.append(t)

    return morning_times, evening_times


def classify_morning(morning_times):
    """
    分类上班打卡
    返回 (status_str, first_morning_min)
    """
    if not morning_times:
        return '上班缺卡', None
    first_t = min(morning_times)  # 最早的一次上班打卡
    if first_t < MORNING_NORMAL_START:
        return '早到', first_t
    elif first_t <= MORNING_NORMAL_END:
        return '正常上班', first_t
    else:
        return '迟到', first_t


def classify_evening(evening_times):
    """
    分类下班打卡
    返回 (status_str, last_evening_min)
    """
    if not evening_times:
        return '下班缺卡', None
    last_t = max(evening_times)  # 最晚的一次下班打卡
    if last_t < EVENING_NORMAL_START:
        return '早退', last_t
    elif last_t <= EVENING_NORMAL_END:
        return '正常下班', last_t
    else:
        return '加班', last_t


def is_time_mode_cell(cell_val):
    """判断一个单元格是否是时间格式（包含 HH:MM 模式）"""
    if not cell_val or not isinstance(cell_val, str):
        return False
    return bool(re.search(r'\d{1,2}:\d{2}', cell_val))


def is_month_time_mode(sheet, header_row, start_row, end_row):
    """判断某个月份的数据块是否为时间模式"""
    time_count = 0
    total_checked = 0
    for r in range(start_row, min(start_row + 3, end_row)):
        name = str(sheet.cell_value(r, 0)).strip()
        if not name or name == '姓名':
            continue
        for c in range(sheet.ncols):
            v = str(sheet.cell_value(r, c)).strip()
            if v and v != '':
                total_checked += 1
                if is_time_mode_cell(v):
                    time_count += 1
    if total_checked > 0:
        return time_count / total_checked > 0.3
    return False


def find_date_header_row(sheet, start_search):
    """从start_search行开始找包含日期数字的表头行"""
    for r in range(start_search, min(start_search + 8, sheet.nrows)):
        digit_cols = 0
        text_markers = 0
        for c in range(6, min(sheet.ncols, 40)):
            v = str(sheet.cell_value(r, c)).strip()
            # 数字 1-31 = 工作日
            if v.isdigit() and 1 <= int(v) <= 31:
                digit_cols += 1
            # 常见休息日标记
            elif v in ('六', '日', '休', '假') or any(kw in v for kw in ['节', '假', '班']):
                text_markers += 1
        if digit_cols >= 5:  # 至少5个工作日列
            return r
        if digit_cols >= 3 and (digit_cols + text_markers) >= 10:  # 混合但有足够列
            return r
    return None


def get_date_columns(sheet, header_row, year, month):
    """从表头行提取日期列映射
    根据中国法定节假日日历，只提取工作日，跳过休息日
    """
    date_cols = {}
    for c in range(6, sheet.ncols):
        v = str(sheet.cell_value(header_row, c)).strip()
        # 数字列 → 通过日历验证是否工作日
        if v.isdigit() and 1 <= int(v) <= 31:
            day = int(v)
            try:
                dt = date_type(year, month, day)
                if is_workday(dt):
                    date_cols[c] = day
            except ValueError:
                pass  # 无效日期（如2月30日）跳过
        # 文本列（六/日/节日）→ 直接跳过

    if date_cols:
        workdays = len(date_cols)
        total = sum(1 for c in range(6, sheet.ncols)
                    if str(sheet.cell_value(header_row, c)).strip())
        print(f"  日期列: {min(date_cols.values())}~{max(date_cols.values())} 日 "
              f"({workdays}/{total} 是工作日)")

    return date_cols


def parse_clock_data(file_path=None, db_path=None):
    """主函数：解析打卡时间数据，构建数据库
    file_path: 可选的Excel文件路径，默认使用内置数据文件
    db_path: 可选的数据库路径，默认使用attendance.db
    """
    if file_path is None:
        file_path = DATA_FILE
    target_db = db_path if db_path is not None else DB_PATH

    print("=" * 60)
    print(f"解析打卡时间数据: {os.path.basename(file_path)}")
    print("废弃考勤状态模式数据，仅使用时间模式")
    print("=" * 60)

    wb = xlrd.open_workbook(file_path)
    sheet = wb.sheet_by_name('考勤表')

    # ===== 1. 识别所有月份块 =====
    month_blocks = []
    for r in range(sheet.nrows):
        v = str(sheet.cell_value(r, 0)).strip()
        if '打卡' in v and ('统计日期' in v or '统计时间' in v):
            m = re.search(r'(\d{4})[-年](\d{1,2})', v)
            if m:
                yr, mo = int(m.group(1)), int(m.group(2))
                hr = find_date_header_row(sheet, r + 1)
                if hr is None:
                    continue
                dc = get_date_columns(sheet, hr, yr, mo)
                if not dc:
                    continue
                month_blocks.append({
                    'year': yr, 'month': mo,
                    'header_row': hr,
                    'start_row': hr + 1,
                    'date_cols': dc,
                })

    # 确定结束行
    for i in range(len(month_blocks)):
        if i + 1 < len(month_blocks):
            month_blocks[i]['end_row'] = month_blocks[i + 1]['header_row']
        else:
            month_blocks[i]['end_row'] = sheet.nrows

    print(f"共识别 {len(month_blocks)} 个月份数据块")

    # ===== 2. 只保留时间模式月份 =====
    time_blocks = []
    for blk in month_blocks:
        if is_month_time_mode(sheet, blk['header_row'], blk['start_row'], blk['end_row']):
            time_blocks.append(blk)

    print(f"其中时间模式: {len(time_blocks)} 个月份")
    for blk in time_blocks:
        # 计算休息日列数
        total_possible = sum(1 for c in range(6, sheet.ncols)
                             if str(sheet.cell_value(blk['header_row'], c)).strip())
        workdays = len(blk['date_cols'])
        restdays = total_possible - workdays
        print(f"  {blk['year']}年{blk['month']:02d}月 ({workdays}工作日/{restdays}休息日)")

    if not time_blocks:
        print("ERROR: 没有找到时间模式数据！")
        return None, None, None

    # ===== 3. 解析打卡记录 =====
    clock_records = []
    for blk in time_blocks:
        yr, mo = blk['year'], blk['month']
        start_row, end_row = blk['start_row'], blk['end_row']
        date_cols = blk['date_cols']

        month_count = 0
        for r in range(start_row, end_row):
            name = str(sheet.cell_value(r, 0)).strip()
            if not name or name in ('', '姓名'):
                continue
            if r > start_row and '打卡' in str(sheet.cell_value(r, 0)):
                break

            for col_idx, day in date_cols.items():
                try:
                    date_obj = datetime(yr, mo, day)
                except:
                    continue

                cell_val = str(sheet.cell_value(r, col_idx)).strip() if sheet.cell_value(r, col_idx) else ''

                # 跳过空单元格和休息日标记
                if not cell_val or cell_val == '':
                    continue
                # 如果单元格内容不含时间数字，跳过（如"休息"、"正常"等状态标记）
                if not is_time_mode_cell(cell_val):
                    continue

                # 按12:00分界解析打卡时间
                morning_times, evening_times = parse_cell_times_split(cell_val)

                # 分类
                morning_cat, first_t = classify_morning(morning_times)
                evening_cat, last_t = classify_evening(evening_times)

                # 格式化时间字符串
                first_str = f'{first_t // 60:02d}:{first_t % 60:02d}' if first_t is not None else ''
                last_str = f'{last_t // 60:02d}:{last_t % 60:02d}' if last_t is not None else ''

                clock_records.append({
                    'name': name,
                    'year': yr,
                    'month': mo,
                    'day': day,
                    'date': date_obj,
                    'first_time_min': first_t,
                    'last_time_min': last_t,
                    'first_time_str': first_str,
                    'last_time_str': last_str,
                    'morning_status': morning_cat,
                    'evening_status': evening_cat,
                })
                month_count += 1

        print(f"  {yr}年{mo:02d}月: {month_count} 条")

    df_clock = pd.DataFrame(clock_records)
    print(f"\n打卡记录总计: {len(df_clock)} 条")
    print(f"  时间范围: {df_clock['date'].min()} ~ {df_clock['date'].max()}")
    print(f"  员工数: {df_clock['name'].nunique()}")

    # ===== 4. 解析花名册 =====
    sheet_roster = wb.sheet_by_name('花名册')
    roster_records = []
    for r in range(sheet_roster.nrows):
        name = str(sheet_roster.cell_value(r, 0)).strip()
        if not name or name in ('姓名', '在职人员', ''):
            continue
        gender = str(sheet_roster.cell_value(r, 1)).strip()
        dept = str(sheet_roster.cell_value(r, 2)).strip().replace('\n', '/')
        position = str(sheet_roster.cell_value(r, 3)).strip()
        rank = str(sheet_roster.cell_value(r, 4)).strip()
        status = str(sheet_roster.cell_value(r, 5)).strip()

        entry_date = None
        ev = sheet_roster.cell_value(r, 8)
        if isinstance(ev, float) and ev > 40000:
            entry_date = datetime(1899, 12, 30) + timedelta(days=int(ev))

        resign_date = None
        resign_reason = ''
        if sheet_roster.ncols > 9:
            rv = sheet_roster.cell_value(r, 9)
            if isinstance(rv, float) and rv > 40000:
                resign_date = datetime(1899, 12, 30) + timedelta(days=int(rv))
        if sheet_roster.ncols > 10:
            resign_reason = str(sheet_roster.cell_value(r, 10)).strip()

        roster_records.append({
            'name': name, 'gender': gender, 'department': dept,
            'position': position, 'rank': rank, 'status': status,
            'entry_date': entry_date, 'resign_date': resign_date,
            'resign_reason': resign_reason,
        })

    df_roster = pd.DataFrame(roster_records)
    print(f"\n花名册: {len(df_roster)} 名员工")
    print(f"  在职: {len(df_roster[df_roster['status'] == '在职'])} 人")
    print(f"  离职: {len(df_roster[df_roster['status'] == '离职'])} 人")

    # ===== 剔除噪声员工及部门 =====
    drop_names = ['员工114', '员工34']
    drop_depts = ['/', '近期离职']
    n_before_roster = len(df_roster)
    n_before_clock = len(df_clock)

    for name in drop_names:
        df_clock = df_clock[df_clock['name'] != name]
    for dept in drop_depts:
        df_roster = df_roster[df_roster['department'] != dept]

    print(f"\n剔除噪声员工及部门:")
    print(f"  花名册: {n_before_roster} -> {len(df_roster)} 人")
    print(f"  打卡记录: {n_before_clock} -> {len(df_clock)} 条")

    # ===== 5. 写入SQLite =====
    if os.path.exists(target_db):
        os.remove(target_db)

    conn = sqlite3.connect(target_db)
    df_clock.to_sql('clock_records', conn, if_exists='replace', index=False)
    df_roster.to_sql('employee_roster', conn, if_exists='replace', index=False)

    conn.execute('CREATE INDEX idx_clock_name ON clock_records(name)')
    conn.execute('CREATE INDEX idx_clock_date ON clock_records(date)')
    conn.execute('CREATE INDEX idx_roster_name ON employee_roster(name)')
    conn.commit()
    conn.close()

    print(f"\n数据库已重建: {target_db}")
    print(f"  clock_records: {len(df_clock)} 行")
    print(f"  employee_roster: {len(df_roster)} 行")

    return df_clock, df_roster


if __name__ == '__main__':
    parse_clock_data()
