"""
生成考勤样例Excel模板（与原始数据格式一致）
格式：日期表头行全部标数字（1~31），与原始Excel完全一致
      系统通过 chinese_calendar 自动识别工作日/休息日
      员工数据行：工作日填打卡时间，休息日留空
"""
import xlwt
import os
import random
from datetime import datetime, date
from chinese_calendar import is_workday


def generate_template():
    wb = xlwt.Workbook(encoding='utf-8')

    # ============================
    # Sheet 1: 花名册
    # ============================
    rs = wb.add_sheet('花名册', cell_overwrite_ok=True)
    headers = ['姓名', '性别', '部门', '岗位', '职级', '员工状态', '员工类型', '归属条线', '进入公司时间']
    for c, h in enumerate(headers):
        rs.write(1, c, h)
        rs.col(c).width = 4000

    sample_data = [
        ['员工A', '男', '财务部', '会计', '初级', '在职', '正式员工', '财务条线', '2023-01-01'],
        ['员工B', '女', '人事部', 'HR', '高级', '在职', '正式员工', '人事条线', '2022-06-01'],
        ['员工C', '男', '研发部', '工程师', '中级', '离职', '正式员工', '技术条线', '2021-03-15'],
        ['员工D', '女', '销售部', '销售经理', '高级', '在职', '正式员工', '销售条线', '2023-09-01'],
        ['员工E', '男', '财务部', '出纳', '初级', '离职', '正式员工', '财务条线', '2022-01-10'],
    ]
    for r, row in enumerate(sample_data):
        for c, v in enumerate(row):
            rs.write(r + 2, c, v)

    # ============================
    # Sheet 2: 考勤表
    # ============================
    cs = wb.add_sheet('考勤表', cell_overwrite_ok=True)

    # 生成2026年1月、2月、3月，与原始格式一致
    months = [
        (2026, 1, '元旦'),
        (2026, 2, '春节'),
        (2026, 3, ''),
    ]

    employees = [
        ('员工A', '财务部', 'UID_A001'),
        ('员工B', '人事部', 'UID_B001'),
        ('员工D', '销售部', 'UID_D001'),
        ('员工C', '研发部', 'UID_C001'),
        ('员工E', '财务部', 'UID_E001'),
    ]

    current_row = 0
    for yr, mo, holiday_name in months:
        import calendar
        days_in_month = calendar.monthrange(yr, mo)[1]

        # 月份标题行（与原始数据格式一致）
        cs.write(current_row, 0, f'打卡时间 统计日期：{yr}-{mo:02d}-01 至 {yr}-{mo:02d}-{days_in_month}')
        current_row += 1

        # 日期表头行：全部标数字（1~31），与原始数据一致
        # 系统会调用 chinese_calendar 自动识别工作日/休息日
        date_headers = ['姓名', '考勤组', '部门', '工号', '职位', 'UserId']

        for d in range(1, days_in_month + 1):
            date_headers.append(str(d))  # 全部标数字，与原始数据格式一致

        for c, h in enumerate(date_headers):
            cs.write(current_row, c, h)
        current_row += 1

        # 员工数据行
        for emp_name, dept, uid in employees:
            cs.write(current_row, 0, emp_name)
            cs.write(current_row, 1, '天津公司')
            cs.write(current_row, 2, dept)
            cs.write(current_row, 5, uid)

            col_idx = 6  # 日期列起始
            for d in range(1, days_in_month + 1):
                dt = date(yr, mo, d)

                # 使用 chinese_calendar 判断工作日（含调休识别）
                if is_workday(dt):
                    # 工作日：写入打卡时间（模拟）
                    random.seed(hash(emp_name + str(yr) + str(mo) + str(d)) % 10000)
                    morn_min = 8 * 60 + random.randint(0, 90)
                    eve_min = 17 * 60 + random.randint(0, 120)
                    morn_h, morn_m = morn_min // 60, morn_min % 60
                    eve_h, eve_m = eve_min // 60, eve_min % 60
                    time_str = f'{morn_h:02d}:{morn_m:02d}  \n{eve_h:02d}:{eve_m:02d}  '
                    cs.write(current_row, col_idx, time_str)

                col_idx += 1

            current_row += 1

        # 月份块之间的空行
        current_row += 1

    out_path = os.path.join(os.path.dirname(__file__), '考勤样例模板.xls')
    wb.save(out_path)
    print(f'样例模板已生成: {out_path}')
    print(f'  员工: 5人, 月份: {[m[0] for m in months]}')


if __name__ == '__main__':
    generate_template()
