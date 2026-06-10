"""
生成符合训练要求的测试数据（全量替换用）
要求：
- 至少 20 名员工，其中至少 2 名离职
- 至少 6 个月数据
- 格式与原始数据一致
"""
import xlwt, os, calendar, random, tempfile
from datetime import date
from chinese_calendar import is_workday

random.seed(42)

OUTPUT = os.path.join(os.path.dirname(__file__), '测试数据_全量替换.xls')

departments = ['技术部', '运营部', '市场部', '财务部', '人事部', '产品部']
positions = {
    '技术部': ['开发','测试','架构师'],
    '运营部': ['运营','客服'],
    '市场部': ['市场','销售'],
    '财务部': ['会计','出纳'],
    '人事部': ['HR','行政'],
    '产品部': ['产品经理','UI设计'],
}

# 生成20+员工
employees = []
resigned_count = 3
# 先加离职员工
for i in range(resigned_count):
    dept = random.choice(departments)
    pos = random.choice(positions[dept])
    employees.append({
        'name': f'离职员工{i+1}',
        'gender': random.choice(['男','女']),
        'dept': dept,
        'position': pos,
        'level': random.choice(['P3','P4','P5','P6']),
        'status': '离职',
        'emp_type': '正式',
        'line': f'{dept}线' if dept != '产品部' else '产品线',
        'join_date': f'202{random.randint(1,3)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}',
        'resign_month': (2025, random.randint(6,9)),  # 2025年中离职
    })
# 再加在职员工
for i in range(20):
    dept = random.choice(departments)
    pos = random.choice(positions[dept])
    employees.append({
        'name': f'员工{i+1}',
        'gender': random.choice(['男','女']),
        'dept': dept,
        'position': pos,
        'level': random.choice(['P3','P4','P5','P6','P7']),
        'status': '在职',
        'emp_type': '正式',
        'line': f'{dept}线' if dept != '产品部' else '产品线',
        'join_date': f'202{random.randint(1,3)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}',
        'resign_month': None,
    })

print(f"生成 {len(employees)} 名员工，其中离职: {resigned_count}")

wb = xlwt.Workbook(encoding='utf-8')

# ===== 花名册 =====
rs = wb.add_sheet('花名册', cell_overwrite_ok=True)
headers = ['姓名','性别','部门','岗位','职级','员工状态','员工类型','归属条线','进入公司时间']
for c, h in enumerate(headers):
    rs.write(0, c, h)
for r, emp in enumerate(employees):
    rs.write(r+1, 0, emp['name'])
    rs.write(r+1, 1, emp['gender'])
    rs.write(r+1, 2, emp['dept'])
    rs.write(r+1, 3, emp['position'])
    rs.write(r+1, 4, emp['level'])
    rs.write(r+1, 5, emp['status'])
    rs.write(r+1, 6, emp['emp_type'])
    rs.write(r+1, 7, emp['line'])
    rs.write(r+1, 8, emp['join_date'])

# ===== 考勤表 =====
cs = wb.add_sheet('考勤表', cell_overwrite_ok=True)
# 生成 2025 年 1 月 ~ 9 月（9个月）
months_data = []
for m in range(1, 10):
    months_data.append((2025, m))

cur_row = 0
for yr, mo in months_data:
    dim = calendar.monthrange(yr, mo)[1]
    
    # 标题行
    cs.write(cur_row, 0, f'打卡时间 统计日期：{yr}-{mo:02d}-01 至 {yr}-{mo:02d}-{dim}')
    cur_row += 1
    
    # 日期表头行（全部标数字）
    hdrs = ['姓名', '考勤组', '部门', '工号', '职位', 'UserId']
    for d in range(1, dim+1):
        hdrs.append(str(d))
    for c, h in enumerate(hdrs):
        cs.write(cur_row, c, h)
    cur_row += 1
    
    # 员工数据
    for emp in employees:
        ename = emp['name']
        # 检查是否已离职
        if emp['resign_month']:
            ry, rmo = emp['resign_month']
            if (yr > ry) or (yr == ry and mo >= rmo):
                continue  # 已离职，跳过
        
        cs.write(cur_row, 0, ename)
        cs.write(cur_row, 1, '天津公司')
        cs.write(cur_row, 2, emp['dept'])
        cs.write(cur_row, 5, f'UID_{ename}')
        
        col = 6
        for d in range(1, dim+1):
            dt = date(yr, mo, d)
            if is_workday(dt):
                seed = hash(ename + str(yr) + str(mo) + str(d)) % 10000
                random.seed(seed)
                # 对不同员工生成不同行为模式
                if emp['status'] == '离职':
                    # 离职员工行为模式：迟到多、加班少
                    mm = 8*60 + random.randint(30, 90)  # 经常迟到
                    em = 17*60 + random.randint(0, 30)  # 很少加班
                else:
                    # 在职员工相对正常
                    mm = 8*60 + random.randint(-10, 60)
                    em = 17*60 + random.randint(0, 90)
                cs.write(cur_row, col, f'{mm//60:02d}:{mm%60:02d}  \n{em//60:02d}:{em%60:02d}  ')
            col += 1
        cur_row += 1
    cur_row += 1  # 月份间空行

wb.save(OUTPUT)
print(f"测试数据已生成: {OUTPUT}")
print(f"  路径: {os.path.abspath(OUTPUT)}")
print(f"  员工: {len(employees)}人 (离职{resigned_count}人)")
print(f"  月份: 2025-01 ~ 2025-09 (9个月)")
print(f"\n使用方式：在左侧菜单「全量替换数据」中上传此文件")
