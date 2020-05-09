# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:06:38 2020

@author: tingshan

main.py
"""
from gurobipy import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

output_path = "output/"
input_set_path = "set/"
input_path = "input/"
demand_path = "demand/"

#讀取資料範圍
year = ['2017']  #年集合
#,'2018'
demand_set = ['123']   # 季集合
#,'456','789','101112'
line_set = ['香腸','條子肉乾']

## 總設定輸入
input_total_set_path = input_set_path  + "總設定.txt"
set_file = open(input_total_set_path, 'r', encoding = 'UTF-8')
set_lines = set_file.readlines()

Cost_day_people = int(set_lines[2].strip("\n")) * 8  # 全職人員日薪
ot_salary = int(set_lines[4]) * 8  # 全職人員加班日薪
Cost_day_short_people = int(set_lines[6].strip("\n")) * 8  # 兼職人員日薪
Cost_people = Cost_day_people * 23  # 全職人員月薪
Cost_short_people = Cost_day_short_people * 23  # 兼職人員月薪
hiring_frequency = int(set_lines[8]) # 兼職人員聘僱頻率(月)
law_rate = float(set_lines[10]) # 每日能來上班比例(根據勞基法)

line_num = int(set_lines[14])  # 生產線數量
#line_set = set_lines[11].strip('\n').split(',')  # 生產線名稱集合



same_machine_num = int(set_lines[17])  # 共用工作站數量
same_machine_name = set_lines[18].strip().split(',')  # 共用工作站名稱
same_m_set = {}  # 共用工作站與各產線內工作站對應
line_no = 19
H = range(1,same_machine_num+1)
for i in range(1, same_machine_num + 1):
    one_line = set_lines[line_no+1 + (i-1) *2].split(';')
    same_m_set[i] = list()
    for one in one_line:
        same_m_set[i].append([int(x) for x in one.split(',')])
same_m_set[1] = [[1,1], [2,1]]
same_m_set[2] = [[2,1]]
same_m_set[3] = []
same_m_set[4] = [[1,3]]

## 生產線設定輸入
work_num = []
mall = {} #需求來源商編號 
day_expiration = {}
Width_d_station = [0] * len(line_set)
Width_station = [0] * len(line_set)
Cost_machine_on = [0]*len(line_set)
Cost_production = [0]*len(line_set)
Yield_rate = {}
P_rate = {}
Capacity_machine = [0]*len(line_set)
Num_machine = [0]*len(line_set)
Cost_inventory = [0]*len(line_set)
People_m_lo = [0]*len(line_set)
Capacity_max = []  # 瓶頸產能
S_stock = []  # 安全庫存量
j = 0
for line in line_set:
    input_procedure_path = "input/設定資料_" + line + ".txt"
    pro_file = open(input_procedure_path, 'r', encoding = 'UTF-8')
    pro_lines = pro_file.readlines()
    work_num += [int(pro_lines[1].strip("\n"))]  # 生產線 j 的工作站總數
    
    line = pro_lines[3].strip("\n").split(";")
    i=0
    for l in line:
        mall[j,i] = [int(x) for x in l.split(",")]  # 生產線 j 的需求來源商
        i=i+1
        
    ex = {}  
    one_line = pro_lines[5].strip("\n").split(";")
    k=0
    for one in one_line:
         ex = [int(x) for x in one.split(',')]  # 允收期限
         day_expiration[j,k] = ex
         k=k+1
                
    Width_d_station[j] = [0]    
    Width_d_station[j].extend([int(x) for x in pro_lines[7].strip("\n").split(",")])  # 生產時間
    Width_station[j] = [0]
    Width_station[j].extend([int(x) for x in pro_lines[9].strip("\n").split(",")])  # 等待時間
    
    Cost_machine_on[j] = [-1]
    Cost_machine_on[j].extend([int(x) for x in pro_lines[11].strip("\n").split(",")])  # 開機成本
    Cost_production[j] = [-1]
    Cost_production[j].extend([int(x) for x in pro_lines[13].strip("\n").split(",")])  # 生產成本
    
    Yield_rate[j] = []
    Yield_rate[j] += [float(x) for x in pro_lines[15].strip("\n").split(",")]  # 製成率
    P_rate[j] = float(1)
    for p in Yield_rate[j]:
        P_rate[j] *= p
    
    Capacity_machine[j] = [-1]
    Capacity_machine[j].extend([int(x) for x in pro_lines[17].strip("\n").split(",")])
    Num_machine[j] = [-1]
    Num_machine[j].extend([int(x) for x in pro_lines[19].strip("\n").split(",")])
    Cost_inventory[j] = [float(x) for x in pro_lines[21].strip("\n").split(",")]
    
    People_m_lo[j] = [-1]
    People_m_lo[j].extend([int(x) for x in pro_lines[23].strip("\n").split(",")]) # 每台機台最少操作人數
    
    S_stock += [int(pro_lines[25]) ] # 安全庫存量
    
    pro_file.close()
    j = j + 1
    
# 計算bottle neck
Capacity_max = []
I = {}
J = set(range(0, len(line_set))) 
for j in J:
    Capacity = 10000000
    I[j] = set(range(1,work_num[j]+1))
    for i in I[j]:
        if(Num_machine[j][i] != 1000): # 如果不是只有人力的話，有機台
            cap = np.prod(np.array(Yield_rate[j][i:]))*Capacity_machine[j][i]*Num_machine[j][i]
            print(cap)
            if(cap < Capacity):
                Capacity = cap
    Capacity_max.append(Capacity)

###############################READ FILES######################################



#############################LEVEL1 FP START###################################
#import model.lv1_fp_t3_0215 as lv1_fp
time_horizon = 50
result_file = open("output/result.txt", 'w')
Capacity_people_rec = {}
Capacity_people = []
for i in line_set:
    Capacity_people_rec[i] = []
#I = {}
virtual_I = {}
for line in range(len(line_set)):
    #result_file.write(line_set[line] + '\n')
    
    # set
    #I[line] = set(range(1,work_num[line]+1)) # 0 是原料1 work_num+1 是最終成品
    virtual_I[line] = set(range(0,work_num[line]+1)) # 0 是原料1 work_num+1 是最終成品
    
    #J = set(range(0, len(line_set))) 
    T = set(range(1, (time_horizon+1)))
       
    machine_open_set = {}
    for i in I[line]:
        for t in T:
            start = t - Width_d_station[line][i]  + 1
            if(start > 1):
                machine_open_set[i,t] = set(range(start,t+ 1))
                
            else:
                machine_open_set[i,t] = set(range(1,t+1))
            machine_open_set[i,t].add(t) # 要加該時間的t
                
    A = {}
    for i in virtual_I[line]:
        for t in T:
            A[i,t] = set()            
                
    # Parameter
    # past inventory
    # 這邊修改了 size 02/18
    Past_inv = {}
    stock = {}
    for i in virtual_I[line]:
        if i == work_num[line]: 
            Past_inv[line,i] = np.random.randint(50, 51, size=max(day_expiration[line,i])) #最終成品庫存
        else:
            Past_inv[line,i] = np.random.randint(50, 51, size=max(day_expiration[line,i])) #在製品庫存
        stock[j,i] = Past_inv[line,i].sum()
    
    
    fp_1_set = range(10,51)
    time_limit = 10
    no = 0
    if no == 0:
        fp_lv1_result = list()
        for r in fp_1_set:
            m = Model('mip_fp_lv1-' )
            o = r
            
            
            # Variable
            # 生產量
            # P = {} # 生產和使用的時間關係
            
            # 生產數
            x = {}
            for i in  virtual_I[line]: # 0 為原物料
                #P[i] = set()
                for k in T:
                    x[i,k] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='x_%d_%d'%(i, k))
            
            # 使用數
            e = {}
            for i in  virtual_I[line]: # 0 為原物料
                #P[i] = set()
                for k in T:
                    e[i,k] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='e_%d_%d'%(i, k))
            
            # 存貨數
            y = {}
            for i in virtual_I[line]:
                y[i,min(T)-1] = sum(Past_inv[line,i])
                for t in T:
                    y[i,t] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='y_%d_%d'%(i, t))
            
            # t時段 開台的台數
            n = {}
            for i in I[line]:
                for t in T:
                    n[i,t] = m.addVar(lb=0, vtype=GRB.INTEGER, name='n_%d_%d'%(i,t))
            
            
            # 聘用的全職機台總人員數
            
            # 於t時段，實際投入機台的人員數(全職)
            p = {}
            for i in I[line]:
                for t in T:
                    p[i,t] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='p_%d_%d'%(i,t))
            
            
            
            ## update
            m.update()
            
            # m.setObjective()設置目標函數
            #m.setObjective((quicksum(x[work_num[line],t] for t in T)), GRB.MAXIMIZE)
            m.setObjective(quicksum(e[work_num[line],t] for t in T), GRB.MAXIMIZE)
            
            # m.addConstr()加入限制式
            #m.addConstrs(((Yield_rate[line][i] * e[i,t]  == x[i+1,min(t+Width_station[line][i]+Width_d_station[line][i],max(T))] ) for t in T for i in (virtual_I[line] - {work_num[line]})) , "工作站製造平衡(2)")
            # 這邊不看等待時間
            m.addConstrs(((Yield_rate[line][i] * e[i,t]  == x[i+1,min(t+1,max(T))] ) for t in T for i in (virtual_I[line] - {work_num[line]})) , "工作站製造平衡(2)")
            m.addConstrs((Capacity_machine[line][i] * n[i,t]  >= x[i,t]  for t in T for i in I[line]),"機台產能平衡(3)") # 這邊好像修改 capacity_ma 的index
            m.addConstrs((quicksum(p[i,t] for i in I[line]) <= o for t in T ),"機台值班（全職）人數應小於聘用人數(4)")
#            m.addConstrs((quicksum( n[i,v]for v in machine_open_set[i,t])<= Num_machine[line][i] for t in T for i in I[line]),"開機機台數小於購買機台數(5)")  # lv 1 FP 不應該限制
            m.addConstrs((quicksum( n[i,v]for v in machine_open_set[i,t])*People_m_lo[line][i] <= p[i,t] for t in T for i in I[line]),"機台聘用人數需要滿足開機時最低人員數(6)")
            m.addConstrs(((x[i,t]+y[i,t-1] == e[i,t] + y[i,t])for i in virtual_I[line] for t in T),"存貨流量平衡(7)")
            # 初始存貨設定(8)直接設定在變數 y 中
            m.addConstrs((x[i,t] == 0 for t in range(min(T),Width_d_station[line][i] + Width_station[line][i]+1) for i in I[line]), "尚未生產前不開工(9)")
               
            m.write("fp_l1-.lp")
            m.setParam('TimeLimit', time_limit)
            m.optimize()
            
            m.write("fp_l1.sol")
            average_production = (m.objVal/o)/time_horizon
            fp_lv1_result.append(average_production)
            result_file.write(str(o) + '\t' + str(average_production) +'\t' + str(m.objVal)+ '\n')
            
    #            fp_lv1_result.append(m.objVal/a.x)
    #        plots = plt.plot(fp_1_set,fp_lv1_result)
    #        plt.savefig(output_pic,dpi = 600)
        Capacity_people_rec[line_set[line]].extend(fp_lv1_result)
        Capacity_people.append(sum(fp_lv1_result[pp] for pp in range(31,41))/10)
        plt.close('all')
        
        [x[work_num[line],t].x for t in T]

        
result_file.close()
#############################LEVEL 1 FP END####################################
"""
MP輸入參數
1. 人均產能: Capacity_people(List)
"""

############################# MP data input ###################################
## 各季資料輸入
Demand = {}
for yy in year:
    for dd in demand_set:
        print("start_mp"+yy+dd)
        input_season_path = demand_path + "demand_basic" + yy + "_" + dd + ".txt"
        season_file = open(input_season_path, 'r', encoding = 'UTF-8')
        season_lines = season_file.readlines()
        
        master_time_horizon = int(season_lines[1])  # 每季規劃期長度
        Cost_material = [-1]
        Cost_material.extend([int(x) for x in season_lines[3].strip("\n").split(",")])  # 原料成本
        
        work_day = [-1]
        work_day.extend([int(x) for x in season_lines[5].strip("\n").split(",")])  # 是否為假日
        
        month_days = []
        month_days.extend([int(x) for x in season_lines[7].strip("\n").split(",")]) # 月份對應天數
        months = range(1, len(month_days)+1)
        
        
        j = 0
        for line in line_set:
            input_demand_path = demand_path + "demand_"+ line + yy + "_" + dd +".txt"
            demand_file = open(input_demand_path, 'r', encoding = 'UTF-8')
            demand_lines = demand_file.readlines()
            
            for m in mall[j,work_num[j]]:
                Demand[j,m] = [-1]
                Demand[j,m].extend([int(x) for x in demand_lines[m + 1].strip("\n").split(",")]) # 成品需求量(t)
                
            j += 1
        
            demand_file.close()
        season_file.close()
############################# MP data input ###################################
        
################################### MP ########################################
#import model.mp_0215.py as mp
    #Capacity_people = [100,100] #fp給定
        P_min = [1000,20] #最低生產量
        
        I = [0]*len(line_set)
        max_wait = [0]*len(line_set)
        #生產_+等待時間
        for j in J:
            max_wait[j]  = int(sum(Width_d_station[j]) + sum(Width_station[j]))
            I[j] = [0, work_num[j]] # 0 是原料，1 是最終成品
        
        #生產時段不用考慮小於生產時間的日期
        prod_T = {}   
        for j in J:
            for i in I[j]:
                if i ==0:
                    prod_T[j,i] = set(range(1, (master_time_horizon+1)))
                else:
                    prod_T[j,i] = set(range(max_wait[j]+1, (master_time_horizon+1)))
        #總規劃時段
        T = set(range(1, (master_time_horizon+1)))
        
        demand_T = set(range(1, master_time_horizon+1))
        time_horizon = len(demand_T)
        
        T_work = set() #開工日集合
        for j in J:
            for i in I[j]:
                for t in prod_T[j,work_num[j]]:
                    if work_day[t] == 0:
                        T_work.add(t)
        
        # 每個月對應的日子
        T_month = {}
        days_count = 0
        for f in range(len(month_days)):
            T_month[f] = range(1, master_time_horizon+1)[days_count: days_count + month_days[f]]
            days_count += month_days[f]
        
        avail_hiring_months = {}
        # 某月可以使用過去哪幾個月聘用的兼職人員
        for f in months:
            avail_hiring_months[f] = range(min(1, f-hiring_frequency+1), f)
        
        #包含從庫存可能來源到規劃最後日期的時段
        virtual_T = {}
        for j in J:
            for i in I[j]:
                for m in mall[j,i]:
                    if i == 0: #原料不考慮生產及等待時間
                        virtual_T[j,i,m] = set(range(1-day_expiration[j,i][m], (time_horizon+1)))
                    else:
                        virtual_T[j,i,m] = set(range(1-max_wait[j]-day_expiration[j,i][m], (time_horizon+1)))
        
        max_expiration = {}
        for j in J:
            for i in I[j]:
                max_expiration[j,i] = max(day_expiration[j,i][m] for m in mall[j,i])
            
        # past inventory
        # 這邊讓 fp 和 mp的 存貨統一
        Past_inv = {}
        stock = {}
        for j in J:
            for i in range(work_num[j]+1):
                if j == 0:
                    if i == 0: #原料不考慮等待時間
                        Past_inv[j,i] = np.random.randint(6000, 6001, size=max_expiration[j,i])
                    elif i == work_num[j]:
                        Past_inv[j,i] = np.random.randint(6000, 6001, size=max_expiration[j,i]+ max_wait[j])
    
                    else:
                        Past_inv[j,i]  = []
                else:
                    if i == 0: #原料不考慮等待時間
                        Past_inv[j,i] = np.random.randint(150, 151, size=max_expiration[j,i])
                    elif i == work_num[j]:
                        Past_inv[j,i] = np.random.randint(150, 151, size=max_expiration[j,i]+ max_wait[j])
                    else:
                        Past_inv[j,i]  = []
    #            stock[i] = Past_inv[i].sum()
        
        past_T = {}
        for j in J:
            for i in I[j]:
                if i == 0:
                    past_T[j,i] = set(range(1-day_expiration[j,i][0], 1))
                else:
                    past_T[j,i] = set(range(1-day_expiration[j,i][0], max_wait[j]+1))
    
        
        K = {} # 可用來源區間    
        
        for j in J:
            for i in I[j]:
                for m in mall[j,i]:
                    for t in past_T[j,i]:
                        K[j,i,t,m] = set()
                    for t in virtual_T[j,i,m]:
    #                    if i == 0:#原料不考慮生產及等待時間
                        k_max = min(time_horizon, (t + day_expiration[j,i][m]))
    #                    else:
    #                        k_max = min(time_horizon, (t + max_wait[j] + day_expiration[j,i][m]))
                        if(t <= 0):
                            K[j,i,t,m] = set(range(1, (k_max+1)))
                        else:
                            K[j,i,t,m] = set(range(t, (k_max+1)))
        
        L = {} #可供給區間
        
        for j in J:
            for i in I[j]:
                for m in mall[j,i]:
                    for t in T:
    #                    if i == 0: #原料不用考慮生產及等待時間
                        k_min = max(-day_expiration[j,i][m]+1, (t - day_expiration[j,i][m]))
    #                    else:
    #                        k_min = max(-max_wait[j]-day_expiration[j,i][m]+1, (t - max_wait[j] - day_expiration[j,i][m]))
    #                    if(t <= max_wait[j] and i == 0) : # 如果不到等待時間，不能使用原料
    #                        L[j,i,t,m] = set(range(k_min , t+1))
    #                    else:
    #                        L[j,i,t,m] = set(range(k_min , t+1))
                        L[j,i,t,m] = set(range(k_min , t+1))
        
        #原料、成品在不同產線/工作站/允收來源所含的過去時段        
       
        P = set() # 生產和使用的時間關係
        A = {}
        for j in J:
           for i in I[j]:
               for m in mall[j,i]:
                   for t in virtual_T[j,i,m]:
                       A[j,i,t,m] = set()
        
        master_m = Model('mip_mp')
        
        # Variable
        
        # 生產量
        x = {}
        
        for j in J:
           for i in I[j]:
               for m in mall[j,i]:
                   for t in T : #
                       for l in L[j,i,t,m]: # 可使用時段
                            x[j,i,l,t,m] = master_m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='x_%d_%d_%d_%d_%d'%(j,i,l,t,m))
                            P.add((l,t))
                            for mm in virtual_T[j,i,m]:
                                if(t > mm and l <= mm):
                                    A[j,i,mm,m].add((l,t))
        
        # 存貨數
        y = {}
        for j in J:
           for i in I[j]:
                for t in T:
                    y[j,i,t] = master_m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='y_%d_%d_%d'%(j, i, t))
            
        # 每日使用的全職人員數
        p = {}
        for j in J:
            for t in prod_T[j,work_num[j]]:
                p[j,t] = master_m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='p_%d_%d'%(j,t))
            for k in range(min(prod_T[j,work_num[j]])-1):
                p[j,k+1] = 0
                
        # 每日使用的兼職人員數
        s = {}
        for j in J:
                for t in prod_T[j,work_num[j]]:
                    s[j,t] = master_m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='s_%d_%d'%(j,t))
        
        # 每月聘用的兼職人員數
        b = {}
        for f in months:
            b[f] = master_m.addVar(lb=0, vtype=GRB.COUNTINUOUS, name='b_%d'%(f))
        
        # 每日是否開工
        o = {}
#        o[0]=0
        for j in J:
            for t in prod_T[j,work_num[j]]:
                o[t] = master_m.addVar(lb=0, vtype=GRB.BINARY, name='o_%d'%(t))
        for t in range(min(max_wait[j] for j in J)):
            o[t+1] = 0
            
        a = master_m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='a_%d'%(t))
        # initial 
        """
        for i in I:
            y[i,0] = stock[i]
        """
#        #規劃過去庫存使用量
#        past_mall_inv = {}
#        for j in J:
#            for i in I[j]:
#                for m in mall[j]:
#                    master_m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='past_mall_inv_%d_%d_%d'%(j,i,m))
        
        #設定規劃安全庫存時間與保存期限有關
        short_T = {}
        for j in J:
            for i in I[j]:
    #            for m in mall[j,i]:
    #                short_T[j,i,m] = range(1,master_time_horizon-day_expiration[j,i][m]+1)
                    short_T[j,i] = range(1,master_time_horizon-max_wait[j]+1)
        
        #原料最早可以供給的成品時段到原料最晚可以生產的時段
        wait_T = {}
        for j in J:
            wait_T[j,work_num[j]] = range(max_wait[j]+1,master_time_horizon-max_wait[j]+1)
    
        total_cost_production = [0]*len(line_set)
        total_cost_machine_on = [0]*len(line_set)
        for j in J:
            total_cost_production[j] = sum(Cost_production[j][1:])
            total_cost_machine_on[j] = sum(Cost_machine_on[j][1:])
        master_m.update()
                    
        # m.setObjective()設置目標函數
        master_m.setObjective((quicksum(x[j,0,t,k,m] * Cost_material[t]  for j in J for t in prod_T[j,0] for m in mall[j,0] for k in K[j,0,t,m])
                       +quicksum(y[j,i,t] * Cost_inventory[j][i] for j in J for i in I[j] for t in T )
                       +quicksum(x[j,work_num[j],t,k,m] * total_cost_production[j] for j in J for t in prod_T[j,work_num[j]] for m in mall[j,work_num[j]] for k in K[j,work_num[j],t,m] )
                       +quicksum(o[t] * total_cost_machine_on[j] for j in J for t in prod_T[j,work_num[j]]) 
                       +a * Cost_people
                       +quicksum(p[j,t]* ot_salary for j in J for t in prod_T[j,work_num[j]]-T_work)
                       +quicksum(b[v]* Cost_short_people for v in avail_hiring_months[f] for f in months )), GRB.MINIMIZE)
        
         
        # m.addConstr()加入限制式
        master_m.addConstrs((P_rate[j] * quicksum(x[j,0,v,t,m] for m in mall[j,0] for v in L[j,0,t,m] )
                            == quicksum(x[j,work_num[j],t+max_wait[j],k,m] for m in mall[j,work_num[j]] for k in K[j,work_num[j],t+max_wait[j],m]) for j in J for t in wait_T[j,work_num[j]] ) , "產線製造平衡(2)")
        master_m.addConstrs((quicksum(x[j,work_num[j],t,k,m] for t in L[j,work_num[j],k,m]) == Demand[j,m][k]  for j in J for k in T for m in mall[j,work_num[j]]),"完成品必須滿足需求(3)")
        master_m.addConstrs(((p[j,t]+s[j,t]) * Capacity_people[j] >= quicksum(x[j,work_num[j],t,k,m] for m in mall[j,work_num[j]] for k in K[j,work_num[j],t,m])for j in J for t in prod_T[j,work_num[j]] ),"當日人員最大產能應大於當日總產量(4)")
        master_m.addConstrs((quicksum(x[j,work_num[j],t,k,m] for m in mall[j,work_num[j]] for k in K[j,work_num[j],t,m])  <= 1*Capacity_max[j] for j in J for t in prod_T[j,work_num[j]] ),"當日產出應小於實際最大產能(5)")
        for j in J:
            for i in I[j]:
                    master_m.addConstrs((y[j,i,t] == quicksum(quicksum(x[j,i,v,k,m] for (v,k) in A[j,i,t,m]) for m in mall[j,i]) for t in T),"存貨計算(6)")
#        master_m.addConstrs((y[j,i,t] == quicksum(Past_inv[j,i][m]-quicksum(x[j,i,l,t,m] for l in L[j,work_num[j],t,m] )) for t in range(1,day_expiration[j,i][m]+1)),"過去庫存存貨計算"    
        
        # 存貨計算(8)(9)已經在變數中直接設定初始化
        master_m.addConstrs((quicksum(x[j,work_num[j],t,k,m] for m in mall[j,work_num[j]] for k in K[j,work_num[j],t,m])<= 99999999*o[t] for j in J for t in prod_T[j,work_num[j]]),"沒開工就沒產量(10)") 
        master_m.addConstrs((quicksum(p[j,t] for j in J) <= a for t in T),"聘僱人數為當日最大全職人數(11)")
        master_m.addConstrs((law_rate * quicksum(b[v] for v in avail_hiring_months[f]) >= quicksum(s[j,t] for j in J) for t in T_month for f in months), "兼職人員使用人數小於當月可用聘雇數")
        master_m.addConstrs((y[j,work_num[j],t] >= S_stock[j] for j in J for t in short_T[j,work_num[j]]),"每日庫存量要大於安全庫存(12)") 
        master_m.addConstrs((quicksum(x[j,work_num[j],t,k,m] for m in mall[j,work_num[j]] for k in K[j,work_num[j],t,m])>= o[t]*P_min[j] for j in J for t in prod_T[j,work_num[j]]),"最低開工產量(13)")
        # 先槓掉，不然會無解orz
        master_m.addConstrs((o[t] == 1 for t in T_work),"最低開工產量(13)")
        
        for j in J:
            for i in I[j]:
                #for m in mall[j,i]:
                past = past_T[j,i]
                master_m.addConstrs((quicksum(quicksum(x[j,i,t,k,mm] for k in K[j,i,t,mm]) for mm in mall[j,i]) <= Past_inv[j,i][t] for t in past),"每個通路的過去存貨平衡(8)(9)")
#                    master_m.addConstrs((quicksum(x[j,i,t,k,m] for k in K[j,i,t,m] ) <=  Past_inv[j,i][t+day_expiration[j,i][m]-1] for t in past),"每個通路的過去存貨平衡(8)(9)")
        
        master_m.write("mp"+ yy + dd+".lp")
        master_m.setParam('TimeLimit', time_limit)
        master_m.optimize() # m.optimize()求解
        master_m.write("mp"+ yy + dd+".sol")
        master_people = {}
        resting_time = []
        for j in J:
            master_people[j] = [int(p[j,t].x) for t in prod_T[j,work_num[j]]] 
        for t in T:
            if t <= min(max_wait[j] for j in J):
                resting_time.append(1-work_day[t]) #規劃外的開工日用平假日決定
            else:
                resting_time.append(int(o[t].x))#規劃內的開工日o
        accurate_hiring = int(a.x)
        
        master_result = {}
        for j in J:
           for i in I[j]:
                   for t in T:
                       for m in mall[j,i]:
                           if(m == mall[j,work_num[j]][0]):
                               master_result[j,i,t] = sum(x[j,i,t,k,m].x for k in K[j,i,t,m] )
                           else:
                               master_result[j,i,t] += sum(x[j,i,t,k,m].x for k in K[j,i,t,m] )
        
################################ MP END #######################################

################################ MP EXCEL輸出(每日生產量及庫存量/全職人數/)  ####
#       master_people = {} #MP不用輸出每日人數(由fp決定)
        
        for j in J:
            file_name = line_set[j]
            mp_output_path =   output_path + file_name + "_" +  yy + "_" +  dd + "_mp_result.xlsx"
            with pd.ExcelWriter(mp_output_path, engine = 'xlsxwriter') as writer: #writer = pd.ExcelWriter(mp_output_path, engine = 'xlsxwriter')
                for i in I[j]:
                    total_need = np.random.randint(0,1,size=(master_time_horizon,len(set(range(1-max(day_expiration[j,i][m] for m in mall[j,i]),master_time_horizon+1)))))
                    for m in mall[j,i]:
                        production_table_row_name = set(range(1-day_expiration[j,i][m],master_time_horizon+1))
                        need = np.random.randint(0,1,size=(master_time_horizon,len(set(production_table_row_name))))
                        for t in T:
                            for k in L[j,i,t,m]:
                                need[t-1][k-min(production_table_row_name)] += x[j,i,k,t,m].x #t時段、通路m的需求量
                                total_need[t-1][k-min(set(range(1-max(day_expiration[j,i]),master_time_horizon+1)))] += x[j,i,k,t,m].x #t時段的總需求量
                        s_df = pd.DataFrame(need,columns = sorted(production_table_row_name),index=T)
                        s_df["SUM"] = s_df.sum(axis=1)
                        s_df["Demand"] = [Demand[j,m][t] for t in T]
    #                    s_df['存貨'] = [y[i,t].x +sum(Demand[1:t+1])*(S_rate-1) for t in T] # 安全庫存應該要被放入模型中的庫存才對
                        s_df.loc["SUM"] = s_df.sum()
                        if i == 0:   
                           s_df.to_excel(writer, sheet_name = "MP＿原料採購時間表_供應商"+str(m+1))
                        else:
                           s_df.to_excel(writer, sheet_name = "MP＿最終成品生產時間表_通路商"+str(m+1))   
                           
                    total_df = pd.DataFrame(total_need,columns = sorted(set(range(1-max(day_expiration[j,i]),master_time_horizon+1))),index=T)
                    total_df["SUM"] = total_df.sum(axis=1)
                    total_df["Demand"] = [sum(Demand[j,m][t] for m in mall[j,i]) for t in T]
                    total_df['存貨'] = [y[j,i,t].x for t in T]
                    total_df.loc["SUM"] = total_df.sum()
                    if i==0 : 
                        total_df.to_excel(writer, sheet_name = "MP＿原料總生產時間表")
                    else:
                        total_df.to_excel(writer, sheet_name = "MP＿最終成品總生產時間表")
                master_df = pd.DataFrame()
                master_df["天數"] = list(prod_T[j,work_num[j]])
                master_df["上班"] = [resting_time[t-1] for t in prod_T[j,work_num[j]]]
                master_df["全職每日聘僱人數"] = master_people[j]
                master_df["兼職每日聘僱人數"] = [s[j,t].x for t in prod_T[j,work_num[j]]]
                master_df["全職總聘僱人數"] = accurate_hiring
                master_df.to_excel(writer, sheet_name = "MP_開工日")
            
                
################################ EXCEL輸出 END ################################
        """
        LV2傳入參數
        1. 每日全職人數: master_people(List)
        2. 全職總人數: accurate_hiring(int)
        3. 開工日: resting_time(List)
        """
############################### LEVEL 2 FP ####################################
    #import model.fp_mp_yt_0217 as lv2_fp
        
    #    writer = pd.ExcelWriter(output_path, engine = 'xlsxwriter')
        
        
    #    for i in I:
    #        production_table_name2 = set(range(1-day_expiration[i],master_time_horizon+1))
    #        need = np.random.randint(0,1,size=(master_time_horizon,len(set(production_table_name2))))
    #        master_result[i] = [-1]
    #        for t in T:
    #            master_result[i].append(0)
    #            for k in K[i,t]:
    #                master_result[i][t] += int(x[i,t,k].x)
    #            for k in L[i,t]:
    #                need[t-1,k-min(production_table_name2)] += x[i,k,t].x
    #        s_df = pd.DataFrame(need,columns = sorted(production_table_name2),index=T)
    #        s_df["SUM"] = s_df.sum(axis=1)
    #        s_df["上班"] = resting_time
    #        s_df["Demand"] = Demand[1:]
    #        s_df['存貨'] = [y[i,t].x  for t in T]
    #        #s_df['存貨'] = [y[i,t].x +sum(Demand[1:t+1])*(S_rate-1) for t in T] # 安全庫存應該要被放入模型中的庫存才對
    #        s_df.loc["SUM"] = s_df.sum()
    #        s_df.to_excel(writer, sheet_name = "生產時間表＿工作站"+str(i))
    #    master_demand.append(master_result[work_num][1:])
    #    
    #    master_df = pd.DataFrame()
    #    master_df["人力"] = master_people
    #    master_df["上班"] = resting_time
    #    master_df.loc["SUM"] = master_df.sum()
    #    master_df.to_excel(writer, sheet_name = "master_result"+str(i))
    #
    #
    #    b1 = np.array(s_df['存貨'][:-1]).reshape(master_time_horizon,1)
    #    b2 = np.array(master_result[work_num][1:]).reshape(master_time_horizon,1)
    #    b3 = np.array(Demand[1:]).reshape(master_time_horizon,1)
    #    b123 = np.concatenate((b1,b2,b3),axis=1)
    #    plots = plt.plot(range(1,master_time_horizon+1),b123)
    #    plt.legend(plots, ["inv","MP_production","demand"], loc='upper right', bbox_to_anchor=(0.8, 1.15),
    #               ncol=(3), framealpha=0.5, prop={'size': 'small', 'family': 'monospace'})
    #    plt.savefig(output_pic_mm,dpi = 600)
    #    plt.close('all') 
        #print(master_result)
        
    
        
        # Set
        #expiration = [20,20,20,20,20]
        get_ratio = 1
        time_range = 30
        if master_time_horizon % time_range == 0: 
            rounds = master_time_horizon // time_range 
        else:
            rounds = master_time_horizon // time_range  + 1
        KC = {}  # dictionary
        KT = {}
        result = {}
    
        fp_p_df = pd.DataFrame()
        fp_inv_df = pd.DataFrame()
        fp_e_df = pd.DataFrame()
        fp_people_df = pd.DataFrame()
        fp_s_df = pd.DataFrame()
        obj_list = list()
        fragment_max = 1
        
        
        print("factory_planning")
        # for r in range(rounds): #只跑一次round
        r = 0
        print("round"+str(r))
        m = Model('mip_fp_lv2-' + dd)
        
        # 多個生產線
        I = {}
        virtual_I = {}
        stock_I = {}
        
        for j in J:
            I[j] = set(range(1,work_num[j]+1))
            virtual_I[j] = set(range(0,work_num[j]+1))
            stock_I[j] = set(range(0,work_num[j]))
        
        #需求來源時段
        demand_T = set(range(r*time_range + max_wait[j] + 1, min((r+1)*time_range+1,master_time_horizon+1)))
        
#        for j in J:
#            for i in I[j]:
#                if i == 0:
#                    prod_T[j,i] = set(range((r-1)*time_range*fragment_max + 1, max(demand_T)*fragment_max+1))
#                else:
#                    prod_T[j,i] = set(range((r-1)*time_range*fragment_max + max_wait[j] + 1, max(demand_T)*fragment_max+1))
                
        T = set(range(r*time_range*fragment_max + 1, (r+1)*time_range*fragment_max+1)) #規劃時段
        f_time_horizon = len(demand_T)
        
        #包含過去存貨的總時段
        virtual_T = {}
        for j in J:
            for i in I[j]:
                for mm in mall[j,i]:
                    virtual_T[j,i,mm] = set(range(1-day_expiration[j,i][mm], (time_range+1)))
        
        #過去存貨可能時段
        past_T = {}
        for j in J:
            for i in I[j]:
                for mm in mall[j,i]:
                    past_T[j,i,mm] = set(range(1-day_expiration[j,i][0], 1))   

#            wait_T = {}
#            for j in J:
#                wait_T[j,work_num[j]] = range(max_wait[j]+1,time_range-max_wait[j]+1)
                
        #對於產線 j 工作站 i 在時間 t 開機到未來的時間
        machine_open_set = {}
        #對於產線 j 工作站 i 所有可能的完成時間
        all_ot_set = {}
        #對於產線 j 工作站 i 在時間 t 投入生產對應的完成時間
        ot_set = {} 
        for j in J:
            for i in virtual_I[j]:
                all_ot_set[j,i] = set()
                for l in T:
                    if(resting_time[int((l-1)/fragment_max)] == 1):
                        if(l+Width_station[j][i]+Width_d_station[j][i] > 0 and l+Width_station[j][i]+Width_d_station[j][i] <= max(T)): #如果完成日>0，則記錄投入生產日及完成日
                            start_k = l #該工作站起始時間
                            k_min = l + Width_d_station[j][i] # 該工作站完成時間
                            
                            need_days = int((k_min-1)/fragment_max) - int((start_k-1)/fragment_max)+1 #完成天( Width_d_station[j][i]/fragment_max+1)
                            restday = need_days - sum(resting_time[int((start_k-1)/fragment_max):int((k_min-1)/fragment_max)+1]) #工作站起始到結束的工作天，如果有假日則
                            
                            print(restday)
                            while(restday > 0):
#                                    restday = 1
                                #start_k = start_k + restday * fragment_max
                                k_min = k_min + restday * fragment_max
                                #print(start_k)
                                #print(k_min)
                                restday = need_days - sum(resting_time[int((start_k-1)/fragment_max):int((k_min-1)/fragment_max)+1])
                                #print(restday)
                                if(need_days  >  sum(resting_time[int((start_k-1)/fragment_max):])): #無法滿足需求
                                    resting_time[int((l-1)/fragment_max)] = 0 #如果該天必定無法滿足需求應該就要設定為不開工
                                    break
                            #k_max = min(time_horizon, (k_min + expiration[i]))
                            end_day = min(k_min+Width_station[j][i]+restday,max(T))
                            ot_set[j,i,l] = {end_day}
                            all_ot_set[j,i].add(end_day)
                            machine_open_set[j,i,l] = set(range(start_k,min(time_range+1,k_min+1)))
                        else:
                            ot_set[j,i,l] = set()
                    else: 
                        ot_set[j,i,l] = set()
                        machine_open_set[j,i,l] ={l}
        
        #對於產線 j 工作站 i 在時間 t 對應到的過去開機時間
        for j in J:
            for i in I[j]:
                for t in T:
                    start = t - Width_d_station[j][i] - Width_station[j][i] + 1
                    if(start > r*fragment_max*time_range):
                        machine_open_set[j,i,t] = set(range(start,t - Width_station[j][i] +1))
                    else: #過去開機時間為上個round規劃
                        #machine_open_set[j,i,t] = set(range(r*fragment_max*f_time_horizon+1,t- Width_station[j][i]+1))
                        machine_open_set[j,i,t] = set()
      
        # Parameter
        Num_machine = [-1,3,8,5,6]
        
        
        
        # Variable
        # 生產量
       # P = {} # 生產和使用的時間關係
        A = {} # 在時段t結束時，其存貨之可能的生產時段與使用時段的配對集合
        for j in J:
            for i in virtual_I[j]:
                for t in T:
                    A[j,i,t] = set()
        # 生產數
        x = {}
        for j in J:
            for i in  virtual_I[j]: # 0 為原物料
                #P[i] = set()
                for t in T:
                    x[j,i,t] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='x_%d_%d_%d'%(j,i,t))
        
        # 使用數
        e = {}
        for j in J:
            for i in  virtual_I[j]: # 0 為原物料
                #P[i] = set()
                for t in T:
                    e[j,i,t] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='e_%d_%d_%d'%(j,i,t))
        
        # 存貨數
        y = {}
        for j in J:
            for i in virtual_I[j]: #最終產品沒有算存貨                    
                y[j,i,min(T)-1] = sum(Past_inv[j,i])
                if i == work_num[j]:
                    y[j,i,t] = [0]
                else:
                    for t in T:
                        y[j,i,t] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='y_%d_%d_%d'%(j,i,t))
                    
        # t時段 開台的台數
        u = {}
        for j in J:
            for i in I[j]:
                for t in T:
                    u[j,i,t] = m.addVar(lb=0, vtype=GRB.INTEGER, name='n_%d_%d_%d'%(j,i,t))
        
        
        # 聘用的全職機台總人員數 + 該期兼職人數
        o = {}
        o = m.addVar(lb=0, vtype=GRB.INTEGER, name='o_%d'%(j))
             
        
        
        # 於t時段，實際投入一般的人員數(全職)
        """
        d = {}
        for i in I:
            for t in T:
                d[i,t] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='d_%d_%d'%(i,t))
        """
        # 於t時段，實際投入機台的人員數(全職)
        p = {}
        for j in J:
            for i in I[j]:
                for t in T:
                    p[j,i,t] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='p_%d_%d_%d'%(j,i,t))
        
        """
        b = {}
        for k in demand_T:
            b[k] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='b_%d'%(k))
        """
        
        # 於t時段，實際投入機台的人員數(兼職)
    
        s = {}
        for j in J:
            for i in I[j]:
                for t in T:
                    s[j,i,t] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='s_%d_%d_%d'%(j,i,t))
        
        
        ## update
        m.update()
        
        
        
        
        # m.setObjective()設置目標函數
        # 這邊不知道一開始參數是用 list還是 dictionary 所以沒有改
        m.setObjective(quicksum ( quicksum(x[j,0,t] * Cost_material[int((t-1)/fragment_max)+1] for t in T)
                       +quicksum(y[j,i,t] * Cost_inventory[j][i] for i in stock_I[j] for t in T)
                       +quicksum(x[j,i,t] * Cost_production[j][i] for i in I[j] for t in T) 
                       +quicksum(u[j,i,t] * Cost_machine_on[j][i] for i in I[j] for t in T)
                       +quicksum(s[j,i,t] * Cost_day_short_people for i in I[j] for t in T)
                       +quicksum(p[j,i,t] * ot_salary   for i in I[j] for t in T-T_work)for j in J)
                       +accurate_hiring * Cost_people, GRB.MINIMIZE) # 人員部分因為全職和兼職合併，無法真實計算成本，建議刪掉
        # 目標是有問題
        
        # m.addConstr()加入限制式
        for j in J:
            m.addConstrs(((x[j,i,t]+y[j,i,t-1] == e[j,i,t] + y[j,i,t])for i in virtual_I[j] for t in T),"存貨流量平衡(2)")
            #m.addConstrs(((x[j,work_num[j],t] == e[j,work_num[j],t]) for t in demand_T),"最終產品不能有存貨")
            for i in I[j]:
                not_all_set = (T-all_ot_set[j,i]) #不開工日
                m.addConstrs((x[j,i,t] == 0 for t in not_all_set),"限制不能生產(11)")
                for t in T:
                    ot_one = ot_set[j,i,t]
                    m.addConstrs(((Yield_rate[j][i-1] * e[j,i-1,t]  == x[j,i,kt] ) for kt in ot_one ) , "工作站製造平衡(1)")

            m.addConstrs((e[j,work_num[j],k] >= master_result[j,work_num[j],k] for k in T),"完成品必須滿足需求")
            m.addConstrs((Capacity_machine[j][i] * u[j,i,t]  >= x[j,i,t] for i in I[j] for t in T ),"機台產能平衡")            
            m.addConstrs((quicksum( u[j,i,v] for v in machine_open_set[j,i,t])*People_m_lo[j][i] <=( p[j,i,t] + s[j,i,t] )for i in I[j] for t in T ),"機台聘用人數需要滿足開機時最低人員數")
            #m.addConstrs((Capacity_people[i]*d[i,t] >= quicksum(x[i,t,k] for k in KC[i, t]) for t in T for i in I ),"人員產能平衡")
            
            """
            for t in demand_T:
                day_set = range((t-1)*fragment_max+1,t*fragment_max+1 )
                m.addConstrs((quicksum(s[i,v] for i in I) <= b[t] for v in day_set ),"機台值班（兼職）人數應小於聘用人數")
            """
            # ranage 需要+1 不然會變空集合
            m.addConstrs((quicksum(x[j,0,kt] for kt in range((k-1)*fragment_max+1,fragment_max*k+1)) >= master_result[j,0,k] for k in demand_T),"原料必須滿足mp規劃的")
            # +1
            #m.addConstrs((x[i,t] == 0 for t in range(min(T),Width_d_station[i] + Width_station[i]+1) for i in I))
            
            # 共用機台  
        m.addConstrs((quicksum(quicksum(u[j-1,i,v]for v in machine_open_set[j-1,i,t]) for j,i in same_m_set[h])<= Num_machine[h] for t in T for h in H),"開機機台數小於購買機台數")
        m.addConstrs((quicksum(p[j,i,t] for j in J for i in I[j]) <= accurate_hiring for t in T ),"機台值班（全職）人數應小於聘用人數")
        # 這一行應該是跟 j 無關，不在for 回圈裡面
        # input 需要對應機台的個數
        
        m.write("fp_l2-" + str(r) + ".lp")
        m.setParam('TimeLimit', time_limit)
        m.optimize() # m.optimize()求解
        m.write("fp_l2-" + str(r) + ".sol")


############################ LEVEL 2 FP END ###################################
################################ FP EXCEL輸出  ################################
    
    for j in J:
        file_name = line_set[j]
        mp_output_path =   output_path + file_name + "_" + yy + "_" +  dd + "_fp_縮小_result.xlsx"
        with pd.ExcelWriter(mp_output_path, engine = 'xlsxwriter') as writer: 
            
            x_df = pd.DataFrame(index=T)
            x_df["天數"] = list(T)
            for i in virtual_I[j]:
                if i == 0:
                    x_df['原料採購'] = [int(x[j,i,t].x)  for t in T]  # 改成老師說的
                else:
                    x_df['w'+str(i)] = [int(x[j,i,t].x)  for t in T]
            x_df.to_excel(writer, sheet_name = "FP採購、生產量時間表")
            
            e_df = pd.DataFrame(index=T)
            e_df["天數"] = list(T)
            for i in virtual_I[j]:
                if i == work_num[j]:
                    e_df['完成品出貨量'] = [int(e[j,i,t].x)  for t in T]
                else:
                    e_df['w'+str(i+1)] = [int(e[j,i,t].x)  for t in T]
                    
                if i == work_num[j]:
                    e_df["master預估需求量"] = [int(master_result[j,i,t])  for t in T]
            e_df.to_excel(writer, sheet_name = "FP使用、出貨量時間表")
            
            
            y_df = pd.DataFrame(index=T)
            y_df["天數"] = list(T)
            for i in virtual_I[j]:
                if i ==0:
                    y_df['原料'] = [int(y[j,i,t].x)  for t in T]
                elif i ==1:
                    y_df['w1-w2'] = [int(y[j,i,t].x)  for t in T]
                elif i ==2:
                    y_df['w2-w3'] = [int(y[j,i,t].x)  for t in T]
                else:
                    y_df['完成品'] = [int(y[j,i,t].x)  for t in T]
            y_df.to_excel(writer, sheet_name = "FP存貨時間表")
            
            n_df = pd.DataFrame(index=T)
            n_df["天數"] = list(T)
            for i in I[j]:
                n_df['w'+str(i)] = [u[j,i,t].x for t in T]
            n_df.to_excel(writer, sheet_name = "FP開機數時間表")
            
            p_df = pd.DataFrame(index=T)
            p_df["天數"] = list(T)
            for i in I[j]:
                p_df['全職_w'+str(i)] = [p[j,i,t].x  for t in T]
            for i in I[j]:
                p_df['兼職_w'+str(i)] = [s[j,i,t].x  for t in T]
            p_df.to_excel(writer, sheet_name = "FP生產時間表_聘僱人員")
    
            
################################ EXCEL輸出 END ################################



