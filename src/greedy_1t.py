#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:50:04 2020

@author: kelly
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 18:23:16 2020

@author: tingshan

greedy.py
"""

from gurobipy import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import copy


big_m = 10000000000000000

# input 檔讀取
demand_set = ['123']  # 季集合

## 總設定輸入
output_path = "output/"
input_set_path = "set/總設定.txt"
set_file = open(input_set_path, 'r', encoding = 'UTF-8')
set_lines = set_file.readlines()

Cost_day_people = int(set_lines[2].strip("\n")) * 8  # 全職人員日薪
#Cost_day_people = 
ot_salary = int(set_lines[4]) * 8  # 全職人員加班日薪
Cost_short_people = int(set_lines[6].strip("\n")) * 8  # 兼職人員日薪
#Cost_short_people = 1
# 全職總薪水，後面再計算

line_num = int(set_lines[10])  # 生產線數量
line_set = set_lines[11].strip('\n').split(',')  # 生產線名稱集合
line_set = ['香腸', '肉乾']  # 暫時設定

same_machine_num = int(set_lines[13])  # 共用工作站數量
same_machine_name = set_lines[14].strip().split(',')  # 共用工作站名稱
same_m_set = {}  # 共用工作站與各產線內工作站對應
line_no = 15
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
input_path = "input/"
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
S_stock = []  # 安全庫存量
max_wait= {} # 生產線一趟最長時長
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
    max_wait[j]  = int(sum(Width_d_station[j]) + sum(Width_station[j]))  # 生產線一趟最長時長
    
    Cost_machine_on[j] = [-1]
    Cost_machine_on[j].extend([int(x) for x in pro_lines[11].strip("\n").split(",")])  # 開機成本
    Cost_production[j] = [-1]
    Cost_production[j].extend([int(x) for x in pro_lines[13].strip("\n").split(",")])  # 生產成本
    
    Yield_rate[j] = []
    Yield_rate[j] += [float(x) for x in pro_lines[15].strip("\n").split(",")]  # 製成率
    P_rate[j] = float(1)  # 生產線總製成率
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
Capacity_min = []
I = {}
J = set(range(0, len(line_set))) 
for j in J:
    Capacity = 10000000
    I[j] = set(range(1,work_num[j]+1))
    for i in I[j]:
        if(Num_machine[j][i] != 1000): # 如果不是只有人力的話，有機台
            cap = np.prod(np.array(Yield_rate[j][i:]))*Capacity_machine[j][i]*Num_machine[j][i]
            if(cap < Capacity):
                Capacity = cap
    Capacity_max.append(Capacity)
    if(j == 0):
        Capacity_min.append(1000)
    else:
        Capacity_min.append(20)

## 各季資料輸入
original_demand = {}
Demand = {}
for dd in demand_set:
    input_season_path = "demand/demand_m" + dd + ".txt"
    season_file = open(input_season_path, 'r', encoding = 'UTF-8')
    season_lines = season_file.readlines()
    
    master_time_horizon = int(season_lines[1])  # 每季規劃期長度
    Cost_material = [-1]
    Cost_material.extend([int(x) for x in season_lines[3].strip("\n").split(",")])  # 原料成本
    
    work_day = [-1]
    work_day.extend([int(x) for x in season_lines[5].strip("\n").split(",")])  # 是否為假日
    
    j = 0
    for line in line_set:
        input_demand_path = "demand/demand_"+ line + dd +".txt"
        demand_file = open(input_demand_path, 'r', encoding = 'UTF-8')
        demand_lines = demand_file.readlines()
        
        for m in mall[j,work_num[j]]:
            original_demand[j,m] = [-1]
            original_demand[j,m].extend([int(x) for x in demand_lines[m + 1].strip("\n").split(",")]) # 成品需求量(t)
            Demand[j,m] = [-1]
            Demand[j,m].extend([int(x) for x in demand_lines[m + 1].strip("\n").split(",")]) # 成品需求量(t)
            # 因為有3個通路，就先把demand除以3
            
        j += 1
    
        demand_file.close()
    season_file.close()

total_cost_production = [0]*len(line_set)
total_cost_machine_on = [0]*len(line_set)
for j in J:
    total_cost_production[j] = sum(Cost_production[j][1:])
    total_cost_machine_on[j] = sum(Cost_machine_on[j][1:])

###############################READ FILES######################################    

############################### GREEDY ########################################

multiplier = np.random.uniform(0, 1, size = master_time_horizon + 1)  # lagrangian multiplier
# 暫時設定全職人員總數
a = {}
a[0] = 5
a[1] = 1

a_set = {}
for j in J:
    a_set[j] = range(a[j],a[j]+31)
 
# 最好的
best_a = {}
for j in J:
    best_a[j] = a[j]

# 最好的
best_s = {}
for j in J:
    best_s[j] = 0
 
# 人均產能
P_avg = [100,30]


# 1t:如果要改成t,k的t是完成的時間（這個時間需段是不是會變成 max_wait~master_time_horizon)
df_all = {}
material_bought_cost = {}
for j in J:  # 每一個產線各跑一次
    expiration = day_expiration[j, work_num[j]][0]  # 成品保存期限
#    df = pd.DataFrame(index = range(1, master_time_horizon + 1))
#    for d_day in range(1, master_time_horizon + 1):  # 需求日
    df = pd.DataFrame(index = range(max_wait[j]+1, master_time_horizon + 1))
    # col:對應的事噓仇日
    # row 是哪天生產的最好
    for d_day in range(max_wait[j]+1, master_time_horizon + 1):  # 需求日
        prod_cost = []  # 每個需求日的單位產能
        #for p_day in range(1, master_time_horizon + 1):  # 成品產出日
        for p_day in range(max_wait[j]+1, master_time_horizon + 1):  # 成品產出日
            cost = big_m # 不能使用的意思
            if p_day <= d_day and (d_day - p_day) <= expiration:
            #if p_day < d_day and (d_day - p_day) <= expiration:  # 比需求日早且在保存期限內
            # 為什麼這邊有 不是小於等於 照理來說應該生產完就可以使用？ 我先改
                cost = 0
                cost += (d_day - p_day) * Cost_inventory[j][-1]  # 成品庫存成本

                
                production_day = p_day - max_wait[j]  # 開始生產日
                    
                # 在可購買原料期間內價格最低的日子購買
                min_material_cost = 100000000
                min_day = -1
                for t in range(production_day - day_expiration[j,0][0], production_day):
                    material_cost = 0
                    if t > 0:
                        material_cost = Cost_material[t]
                        
                    material_cost += Cost_inventory[j][0] * (d_day - t - max_wait[j])
                    
                    if material_cost < min_material_cost:
                        min_material_cost = material_cost
                        min_day = t
                material_bought_cost[j, d_day] = [min_day, min_material_cost]  # 紀錄哪天的原料在哪天購買&價格
                cost += min_material_cost  # 原料購買成本
                    
                if production_day > 0:  # 不是過去存貨才加入人員成本
                    if work_day[production_day] == 0:
                        cost += multiplier[production_day] / P_avg[j]
                    else:
                        cost += ot_salary * multiplier[production_day] / P_avg[j]
            prod_cost.append(cost)
            
        df[d_day] = prod_cost
    df_all[j] = df

# 過去存貨
I = {}
Past_inv = {} # j：生產線,i：工作站
stock = {} 
for j in J: 
    for i in range(work_num[j]+1):
        expiration = day_expiration[j,i][0]
        if j == 0:  # 香腸初始存貨
            if(i in {0, work_num[j]}):  # 是原料或成品
                if(i == 0):  # 原料過去存貨
                    Past_inv[j,i] = np.random.randint(10000, 10001, size=expiration)
                else:  # 成品過去存貨
                    Past_inv[j,i] = np.random.randint(10000, 10001, size=expiration+ max_wait[j])
            else:
                Past_inv[j,i]  = []
        else:  # 肉乾初始存貨
            if(i in {0, work_num[j]}):
                if(i == 0):
                    Past_inv[j,i] = np.random.randint(200, 201, size=expiration)
                else:
                    Past_inv[j,i] = np.random.randint(50, 51, size=expiration)

            else:
                Past_inv[j,i]  = []
    I[j] = [0, work_num[j]]
    
# 需要考慮mall的問題～
# material 在t時間生產 在對應b_t時購買會有最小成本


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


virtual_T = {}
for j in J:
    for i in I[j]:
        for m in mall[j, i]:
            virtual_T[j,i,m] = set(range(1 - day_expiration[j,i][m], (master_time_horizon + 1)))

T_work = {} #開工日集合
for j in J:
    T_work[j] = set()
    for t in prod_T[j,work_num[j]]:
        if work_day[t] == 0:
            T_work[j].add(t)
Cost_people = Cost_day_people * len(set(T_work) - set(range(1,max_wait[j]+1) )) # 全職人員月薪

short_T = {}
for j in J:
    for i in I[j]:
        for m in mall[j,i]:
            short_T[j,i,m] = range(1,master_time_horizon-day_expiration[j,i][m]+1)

# K,L 還需要改，需要和梓妘對應
K = {} # 可用來源區間    
for j in J:
    for i in I[j]:
        for m in mall[j,i]:
            for t in virtual_T[j,i,m]:
                k_max = min(master_time_horizon, (t + day_expiration[j,i][m]))
                if(t <= 0):
                    K[j,i,t,m] = set(range(1, (k_max+1)))
                else:
                    K[j,i,t,m] = set(range(t, (k_max+1)))

L = {} #可供給區間
for j in J:
    for i in I[j]:
        for m in mall[j,i]:
            for t in T:
                k_min = max(-day_expiration[j,i][m]+1, (t - day_expiration[j,i][m]))
                L[j,i,t,m] = set(range(k_min , t+1))
P = set() # 生產和使用的時間關係
A = {}
for j in J:
   for i in I[j]:
       for m in mall[j, i]:
           for t in virtual_T[j,i,m]:
               A[j,i,t,m] = set()

   

for j in J:
   for i in I[j]:
       for m in mall[j, i]:
           for t in T:
               for l in L[j,i,t,m]: # 可使用時段
                    P.add((l,t))
                    for mm in virtual_T[j,i,m]:
                        if(t > mm and l <= mm):
                            A[j,i,mm,m].add((l,t))

best_production_table = {}
production_table = {}
best_product_sum = {}
product_sum = {}
for j in J:
    best_production_table[j] = {}
    production_table[j] = {}
    best_product_sum[j] = {}
    product_sum[j] = {}
    



    
real_cost = {}
min_cost = {}
# 全職人員使用數
p = {} # 整數？ 先設小數
s = {}
best_s = {}
best_p = {}
o = {}
y = {}
best_y = {}
for j in J:
    p[j] = {}
    s[j] = {}
    best_s = {}
    best_p = {}
    y [j] = {}
    best_y[j] = {}
    real_cost[j] = {}
    min_cost[j] = big_m*1000
    for t in T:
        p[j][t] = 0
        s[j][t] = 0



        

for j in J:
    print("生產線 "+str(j))
    for temp_a in a_set[j]:
        print("全職人數：" + str(temp_a))
        # initialize
        a[j] = temp_a
        for t in T:
            p[j][t] = 0
            s[j][t] = 0
        
        for i in I[j]:
            for m in mall[j,i]:
                for t in virtual_T[j,i,m]: 
                    for b_t in K[j,i,t,m]:
                        production_table[j][i,t,b_t,m] = 0
        for i in I[j]:
            for t in virtual_T[j,i,0]: 
                product_sum[j][i,t] = 0
        
        Demand = copy.deepcopy(original_demand)
         
        # 先用存貨滿足需求量
        print("先用存貨滿足需求量")
        for m in sorted(mall[j, work_num[j]],reverse = True): # 從允收期小的開始給
            for t in range(-(day_expiration[j, work_num[j]][m])+1,max_wait[j]+1):
                #print(Past_inv[j,work_num[j]][t])
                rest = Past_inv[j,work_num[j]][t]  # 成品在第t天的存貨剩餘量
                if(t < 1):
                    d_day = 1
                else: # 預先後面會生產
                    d_day = t
                while(rest != 0):
    #                print(rest)
                    while(Demand[j,m][d_day] == 0):  # 如果成品需求被滿足就跳下一天
                        d_day += 1
                        if(d_day not in K[j,work_num[j],t,m]):
                            break
                    if(d_day not in  K[j,work_num[j],t,m]):  # 成品不在保存期限內就不要
                        break
                    #print(Demand[j,m][d_day])
                    max_add = min(Demand[j,m][d_day],rest)
                    rest -= max_add
                    production_table[j][work_num[j],t,d_day,m] += max_add
                    product_sum[j][work_num[j],t] += max_add
                    Demand[j,m][d_day] -= max_add
            # 最小開工產量
            # 次要目標是最低存貨 -> 對應的原料也需要跟著購買
        print("最小開工產量")
        for t in T_work[j]:
            # 這部分在前面做處理
            #if(t > max_wait[j]): # 前五天只能購買原料
            
            rest = Capacity_min[j]
            
            p[j][t] += Capacity_min[j]/P_avg[j]
            product_sum[j][work_num[j],t] += Capacity_min[j]
        #        print(p[j,t])
        #        print(product_sum[j,work_num[j],t])
            m = max(mall[j,work_num[j]])
            while(rest != 0):
                #print(rest)
                d_day = min(t+max_wait[j],master_time_horizon)
                while(Demand[j,m][d_day] == 0):
                    if(m != 0 ):
                        m -= 1
                    else:
                        d_day += 1
                        m = max(mall[j,work_num[j]])
                    if(d_day > master_time_horizon):
                        break
                if(d_day not in K[j,work_num[j],t,m]):
                    break
                #print(Demand[j,m][d_day])
                max_add = max(Demand[j,m][d_day],rest)
                rest -= max_add
                production_table[j][work_num[j],t,d_day,m] += max_add
                Demand[j,m][d_day] -= max_add

                    
        print("依序安排生產量")
        # 依序安排生產量
        #print("依序安排生產量")
        # 最小成本
        sort_demand_schdule = sorted(df_all[j].columns,key = lambda x:df_all[j].min()[x],reverse = True) # i
#==============================================================================
#         other option
#           ii. 應該要最小的和次小的差
#           iii. 最小的幾個平均
#           
#        min_cost_l = {}
#        min_num = 10
#        for t in df_all[j].columns:
#            sorted_cost = sorted(df_all[j][t].index, key = lambda x: df_all[j][t][x])
#            min_cost_l[t] = df_all[j][t][sorted_cost[1]] - df_all[j][t][sorted_cost[0]] # ii
#           # min_cost_l[t] = sum(df_all[j][t][sorted_cost[m]] for m in range(min_num ))/min_num # iii
#        sort_demand_schdule = sorted(df_all[j].columns,key = lambda x:min_cost_l[x],reverse = True)
#==============================================================================
        sort_demand_schdule = [x for x in sort_demand_schdule if x > max_wait[j]]
        for t in sort_demand_schdule:
            #print("t "+str(t))
            c_df = df_all[j][t] # cost的dataframe
            cost_day_sort = sorted(c_df.index,key = lambda x:c_df[x])
            for m in sorted(mall[j, work_num[j]],reverse = True): # 從允收期小的開始給
                no_d = max_wait[j]+1 # T3:從第一天開始而不是第0天? 1t:因為c_df是從1開始～
                add_days = 1
                cost_day_sort_m = [x for x in cost_day_sort if x in L[j,work_num[j],t,m]]
                b_t = cost_day_sort_m[no_d-max_wait[j]-1]
                while(Demand[j,m][t] > 0):
                    """
                    print(str(j)+' ' + str(m) + ' ' + str(t))
                    print(Demand[j,m][t])
                    print("b_t " +str(b_t))
                    print()
                    """
                    #rest = Capacity_max[j] - product_sum[j,work_num[j],b_t]
                    #if(rest >= Demand[j,m][t]): # 當天還可以生產
                    if(a[j]*P_avg[j]-product_sum[j][work_num[j],b_t] >= Demand[j,m][t]):# 如果全職全部都生產的產能大於需求
                        p[j][b_t] += Demand[j,m][t]/P_avg[j] # 這部分不知道要不要把max_wait的天數都加？
                        
                        """
                        print(p[j,b_t])
                        print("如果全職全部都生產的產能大於需求")
                        print(a[j]*P_avg[j])
                        print(product_sum[j,work_num[j],b_t])
                        print(Demand[j,m][t])
                        """
                        production_table[j][work_num[j],b_t,t,m] += Demand[j,m][t]
                        product_sum[j][work_num[j],b_t] += Demand[j,m][t]
                        Demand[j,m][t] = 0
                    else: # 如果沒有的話，就要考慮換天還是兼職
                        """
                        print("如果沒有的話，就要考慮換天還是兼職")
                        # 先把當天全職用掉
                        print(P_avg[j] * (a[j]-p[j,b_t]))
                        print(product_sum[j,work_num[j],b_t])
                        """
                        Demand[j,m][t] -= P_avg[j] * (a[j]-p[j][b_t])
                        production_table[j][work_num[j],b_t,t,m] += P_avg[j] * (a[j]-p[j][b_t])
                        product_sum[j][work_num[j],b_t] += P_avg[j] * (a[j]-p[j][b_t])
                        p[j][b_t] = a[j]
                        #print(p[j,b_t])
                        other_rest = a[j]*P_avg[j] - product_sum[j][work_num[j],no_d+add_days]
                        while(other_rest == 0):  # 如果過去其他天剩餘可提供給今天的產能
#                            print(add_days)
                            add_days += 1
                            other_rest = a[j]*P_avg[j] - product_sum[j][work_num[j],no_d+add_days]
                        max_add = max(Capacity_max[j] - (s[j][b_t]+a[j]) * P_avg[j],0)
                        if(no_d-max_wait[j]-1 < len(cost_day_sort_m) - 1):
                            if(c_df[no_d] + (Cost_short_people)/P_avg[j]  # 使用兼職
                               <= c_df[cost_day_sort_m[no_d-max_wait[j]-1]] ):
                                if(max_add >= Demand[j,m][t]):
                                    print(str(t)+" 1:"+ str(Demand[j,m][t] / P_avg[j]))
                                    s[j][b_t] += Demand[j,m][t] / P_avg[j]
                                    production_table[j][work_num[j],b_t,t,m] += Demand[j,m][t]
                                    product_sum[j][work_num[j],b_t] += Demand[j,m][t]
                                    Demand[j,m][t] = 0
                                else: # 沒有考慮其他天的兼職（可能有的天成本更低，但只有開全職，無法追溯成本更低的天數）
                                    print(str(t)+" 2:"+ str(max_add / P_avg[j]))
                                    s[j][b_t] += max_add / P_avg[j]
                                    production_table[j][work_num[j],b_t,t,m] += max_add
                                    product_sum[j][work_num[j],b_t] += max_add
                                    Demand[j,m][t] -= max_add
                                    no_d += 1
                            else: # 使用另一天
                                no_d += add_days
                        else:
                            if(max_add >= Demand[j,m][t]):
                                print(str(t)+" 3:"+ str(Demand[j,m][t] / P_avg[j]))
                                s[j][b_t] += Demand[j,m][t] / P_avg[j]
                                production_table[j][work_num[j],b_t,t,m] += Demand[j,m][t]
                                product_sum[j][work_num[j],b_t] += Demand[j,m][t]
                                Demand[j,m][t] = 0
                            else: # 沒有考慮其他天的兼職（可能有的天成本更低，但只有開全職，無法追溯成本更低的天數）
                                print(str(t)+" 4:"+ str(max_add / P_avg[j]))
                                s[j][b_t] += max_add / P_avg[j]
                                production_table[j][work_num[j],b_t,t,m] += max_add
                                product_sum[j][work_num[j],b_t] += max_add
                                Demand[j,m][t] -= max_add
                                no_d += 1
                        
    
                        if(no_d-max_wait[j]-1 >= len(cost_day_sort_m)): #兼職
                            #print("無法滿足的需求")
                            #print(str(j)+' ' + str(m) + ' ' + str(t))
                            #print(Demand[j,m][t])
                            for s_t in cost_day_sort_m:
                                max_s_num =  Capacity_max[j]/P_avg[j] - a[j] - s[j][s_t]
                                if(max_s_num > 0):
                                    if(Demand[j,m][t] <= max_s_num * P_avg[j]): # s_t 當天兼職能滿足剩餘Demand
                                        print(str(t)+" 5:"+ str(Demand[j,m][t] / P_avg[j]))
                                        s[j][s_t] += Demand[j,m][t] / P_avg[j]
                                        production_table[j][work_num[j],b_t,t,m] += Demand[j,m][t]
                                        product_sum[j][work_num[j],b_t] += Demand[j,m][t]
                                        Demand[j,m][t] = 0
                                        break
                                    else:
                                        print(str(t)+" 6:"+ str(max_s_num))
                                        s[j][s_t] += max_s_num
                                        production_table[j][work_num[j],b_t,t,m] += max_s_num * P_avg[j]
                                        product_sum[j][work_num[j],b_t] += max_s_num * P_avg[j]
                                        Demand[j,m][t] -= max_s_num * P_avg[j]
                            break
                        else:
                            #print("!")
                            b_t = cost_day_sort_m[no_d-max_wait[j]-1]
                            
        
    
        print("原料存貨")
        # 用存貨減去
        for m in sorted(mall[j, 0],reverse = True): # 從保存期限期小的開始給
            for t in range(-(day_expiration[j, 0][m])+1,1):
                rest = Past_inv[j,0][t]  # 成品在第t天的存貨剩餘量
                use_set = sorted(list(K[j,0,t,m]),reverse = True)
                #print(t)
                #print(use_set)
                material_times = [b_t for k in use_set for b_t in range(1,max(L[j,0,max(use_set),m])+1) if production_table[j][0,b_t,k,m] != 0]
                sort_material_times = sorted(material_times,key = lambda i :material_bought_cost[j,i],reverse = True)
                #sorted(use_set,key = lambda i : )
                d_day = 0
                m_day = 0
                if(len(sort_material_times) != 0 ): # 有使用存貨的可能性
                    while(rest != 0):
        #                print(rest)
                        while(production_table[j][0,sort_material_times[m_day],use_set[d_day],m] == 0):  # 如果成品需求被滿足就跳下一天
                            d_day += 1
                            if(d_day >= len(use_set)):
                                m_day += 1
                                d_day = 0
                                if(m_day >= len(sort_material_times)):
                                    break
                        if(m_day >= len(sort_material_times)):  
                            break
                        #print(Demand[j,m][d_day])
                        if(rest >= production_table[j][0,sort_material_times[m_day],use_set[d_day],m]):  # 存貨滿足 demand
                            rest -= production_table[j][0,sort_material_times[m_day],use_set[d_day],m]
                            production_table[j][0,t,use_set[d_day],m] += production_table[j][0,sort_material_times[m_day],use_set[d_day],m]
                            product_sum[j][0,t] += production_table[j][0,sort_material_times[m_day],use_set[d_day],m]
                            product_sum[j][0,sort_material_times[m_day]] -=  production_table[j][0,sort_material_times[m_day],use_set[d_day],m]
                            production_table[j][0,sort_material_times[m_day],use_set[d_day],m] = 0
                            
                        else:
                            production_table[j][0,sort_material_times[m_day],use_set[d_day],m] -= rest  # 存貨滿足部分 demand
                            product_sum[j][0,sort_material_times[m_day]] -= rest  # 存貨滿足部分 demand
                            production_table[j][0,t,use_set[d_day],m] += rest
                            product_sum[j][0,t] += rest
                            rest = 0
        
        # y[j,i,t] == quicksum(quicksum(x[j,i,v,k,m] for (v,k) in A[j,i,t,m]) for m in mall[j]
        # 存貨計算
        y[j] = {} # initialize
        for i in I[j]:
            for t in T:
                y[j][i,t] = 0
        
        # 計算成品庫存
        i = work_num[j]
        for t in T:
            for m in mall[j, i]:
                if i == 0:
                    l = t-1
                else:
                    l = t
                for v,k in A[j,i,l,m]:
                    y[j][i,t] += production_table[j][i,v,k,m]
        print("安全庫存檢查")
        # 安全庫存檢查
        # 目前沒有考慮原料是否需要重新購買... 要的話，可以把原料存貨計算放在最後面即可
        #short_T = range(1,80)
        
        k = 0
        not_suit_ss = [t for t in short_T[j,i,m] if y[j][work_num[j],t] < S_stock[j]]
        while k < len(not_suit_ss):
            short_exp_m = mall[j,work_num[j]][-1]
            rest = S_stock[j] - y[j][work_num[j],t]
            k_day = not_suit_ss[k] + 1 # 原生產的日期
            d_day = 0 # 有需求的index（對應 demand_set）
            m_day = k_day # 為了有安全庫存而生產的日子
            while(rest > 0):
                while(product_sum[j][work_num[j],m_day] >= Capacity_max[j]): # 需要有適當的產能 -> 生產存貨
                    m_day -= 1 # 往前去尋找還有產能的日子
                demand_set = list(K[j,work_num[j],m_day,short_exp_m]) # m_day 對應的生產日期
                demand_set = [d for d in demand_set if d in K[j,work_num[j],k_day,short_exp_m]] # 需要扣除掉由 k_day 無法使用的日期
                while(production_table[j][work_num[j],k_day,demand_set[d_day],short_exp_m] <= 0): 
                    # 依序去看過去規劃中由 k_day 生產滿足d_day 是否有值，若是沒有，去看下一個需求天
                    # T3: 這邊為什麼不是從k_day+1開始看生產的d_day是否有值?
                    d_day += 1 
                    if(d_day >= len(demand_set)): # 若是對應的需求日都沒有值，則看由下一天看的
                        k_day += 1
                        d_day = 0 # 由於需求的改變，使用必會晚一點
                        demand_set = [t for t in demand_set if t >= k_day]
                max_add = min(rest,production_table[j][work_num[j],k_day,demand_set[d_day],short_exp_m]) # 最多能滿足的存貨量
                rest -= max_add
                production_table[j][work_num[j],k_day,demand_set[d_day],short_exp_m] -= max_add
                product_sum[j][work_num[j],k_day] -= max_add
                # 人員異動
                people_minus = max_add/P_avg[j]
                minus_p = max(0,s[j][k_day] - people_minus)
                p[j][k_day] -= people_minus -( s[j][k_day] - minus_p)
                s[j][k_day] = minus_p
                 
                # 要哪天生產
                make_rest = max_add # 對應要多生產的數量
                while(make_rest > 0): # 沒有重新考慮保存期限的問題
                    max_production = min(max_add ,Capacity_max[j]-product_sum[j][work_num[j],m_day]) # m_day 天最多能多增加的生產量
                    production_table[j][work_num[j],m_day,demand_set[d_day],short_exp_m] += max_production
                    product_sum[j][work_num[j],m_day] += max_production
                    # 人員異動
                    people_add = max_production/P_avg[j]
                    add_p = max(a[j],p[j][m_day]+people_add) - p[j][m_day]
                    s[j][m_day] += people_add - (add_p - p[j][m_day])
                    p[j][m_day] = add_p
                    make_rest -= max_production
                    m_day -= 1

                """
                保存期限問題
				if m_day - d_day < day_expiration[j,work_num[j]][short_exp_m]:
					做上面那些事
				else: # 兩種做法?
					1. 直接忽略，使得安全存貨不一定被滿足(soft constraint)
					2. 多生產，讓m天一樣生產min(rest, capacity_max[j] - product_sum[j][work_num[j],m_day])
					   只是生產出來的量就會一路放到m+day_expiration[j,work_num[j][short_exp_m]
             	"""
        #==============================================================================
        #     buy material
        #     初始原料存貨怎麼辦？
        #     1. 從最早的存貨開始用，並代替其使用時間距離最遠的部分
        #     2. 使用其使用範圍內最貴購買原料期間（暫定）
        #==============================================================================
        
        
        for t in range(max_wait[j]+1,master_time_horizon+1):
            b_t = material_bought_cost[j,t][0] # 考慮開始生產時的原料購買成本，原本計算就是依照完成日去看，但好想有點問題
            production_table[j][0,b_t,t-max_wait[j],0] = product_sum[j][work_num[j],t]/P_rate[j] # 使用時間為使用的前五天
            product_sum[j][0,b_t] += product_sum[j][work_num[j],t]/P_rate[j]
        
        
        # 計算原料庫存
        i = 0
        for t in T:
            for m in mall[j, i]:
                if i == 0:
                    l = t-1
                else:
                    l = t
                for v,k in A[j,i,l,m]:
                    y[j][i,t] += production_table[j][i,v,k,m]
                    
            
            
        
        # 是否有開工
        o = {}
        for t in T:
            if product_sum[j][work_num[j],t] > 0 :
                o[t] = 1
            else:
                o[t] = 0
        
    #==============================================================================
    #     master_m.setObjective((quicksum(x[j,0,t,k,m] * Cost_material[t]  for j in J for t in T for m in mall[j,0] for k in K[j,0,t,m])
    #                    +quicksum(y[j,i,t] * Cost_inventory[j][i] for j in J for i in I[j] for t in T )
    #                    +quicksum(x[j,work_num[j],t,k,m] * total_cost_production[j] for j in J for t in T for m in mall[j,work_num[j]] for k in K[j,work_num[j],t,m] )
    #                    +quicksum(o[t] * total_cost_machine_on[j] for t in T for j in J) 
    #                    +a * Cost_people
    #                    +quicksum(p[j,t]* ot_salary for j in J for t in T-T_work) ## T不對
    #                    +quicksum(s[j,t]* Cost_short_people for j in J for t in T )), GRB.MINIMIZE)
    #==============================================================================
    
        total_cost = sum(Cost_material[t] * product_sum[j][0,t] for t in T )
        total_cost += sum(Cost_inventory[j][i] * y[j][i,t] for i in I[j] for t in T)
        total_cost += sum(total_cost_production[j] * product_sum[j][work_num[j],t] for t in T )
        total_cost += sum(total_cost_machine_on[j] * o[t] for t in T)
        total_cost += a[j] * Cost_people
        total_cost += sum(ot_salary * p[j][t] for t in set(T)-set(T_work[j]))
        total_cost += sum(s[j][t]* Cost_short_people for t in T)
        real_cost[j][a[j]] = total_cost
        total_cost += sum(multiplier[t] * p[j][t] for t in T) # 這個不知道要不要加，還是要另計（為了largange應該要加入，但最終要消失..)
        print(total_cost)
        print(str(a[j])+"short_time:"+str(sum(s[j][t] for t in T)))
        if(total_cost < min_cost[j]):
            print("changed:"+str(a[j]))
            min_cost[j] = total_cost
            
            best_a[j] = a[j]
            best_production_table[j] = production_table[j].copy()
            best_p[j] = copy.deepcopy(p[j])
            best_s[j] = copy.deepcopy(s[j])
            best_product_sum[j] = copy.deepcopy(product_sum[j])
            best_y[j] = copy.deepcopy(y[j])
for j in J:
    print(best_a[j])
#==============================================================================
# 可能的 bug
# 1. productsum 和 producttable不一致：已檢查（沒問題）
# 2. 兼職人員沒有限制到：已修正
# 3. 有一些Demand 沒全滿足（肉乾）：解決
# 4. 生產量與人力配置沒有一致：解決
# 5. 在安排最低生產量時，要算人力：解決
# 6. 最低生產量從小的開始：解決
# 7. 原料存貨的使用：半解決（確認是否是真的不能使用）
# 8. 總成本的計算：半解決(mutiplier的問題)
# 9. o[t]還沒有計算：解決 
# 10. 安全庫存：半解決（確認是否是真的不能使用）
#==============================================================================

    # output 
master_result = {}
    
for j in J:
    file_name = line_set[j]
    mp_output_path =   output_path + file_name+"_greedy_result.xlsx"
    with pd.ExcelWriter(mp_output_path, engine = 'xlsxwriter') as writer: #writer = pd.ExcelWriter(mp_output_path, engine = 'xlsxwriter')
        for i in I[j]:
            for m in mall[j, i]:
                if i == 0: #原料不考慮生產及等待時間
                    production_table_row_name = set(range(1-day_expiration[j,i][m],master_time_horizon+1))
                else:
                    production_table_row_name = set(range(1-day_expiration[j,i][m]-max_wait[j],master_time_horizon+1))
                need = np.random.randint(0,1,size=(master_time_horizon,len(set(production_table_row_name))))
                #master_result[i] = [-1]
                for t in T:
#                        master_result[i].append(0)
#                        for k in K[j,i,t,m]:
#                            master_result[i,t] += int(x[j,i,t,k,m].x) #t時段的生產量
#                            need[t-1][k-min(production_table_row_name)] = x[j,i,k,t,m].x #t時段的需求量
                    for k in L[j,i,t,m]:
                        need[t-1][k-min(production_table_row_name)] += best_production_table[j][i,k,t,m] #t時段的需求量
#                s_df = s_df.transpose()
                s_df = pd.DataFrame(need,columns = sorted(production_table_row_name),index=T)
                s_df["SUM"] = s_df.sum(axis=1)
                s_df["Demand"] = [original_demand[j,m][t] for t in T]
                s_df['存貨'] = [best_y[j][i,t]  for t in T]       
                s_df['全職人員數'] = [best_p[j][t] for t in T]
                s_df["兼職人員數"] = [best_s[j][t] for t in T]
#                s_df['存貨'] = [y[i,t].x +sum(Demand[1:t+1])*(S_rate-1) for t in T] # 安全庫存應該要被放入模型中的庫存才對
                s_df.loc["SUM"] = s_df.sum()
                if i==0:   
                    s_df.to_excel(writer, sheet_name = "MP_greedy＿原料採購時間表_來源商"+str(m+1))
                else:
                    s_df.to_excel(writer, sheet_name = "MP_greedy＿最終成品生產時間表_來源商"\
                                  +str(m+1)+"允收"+str(day_expiration[j,work_num[j]][m]))
        


            need = np.random.randint(0,1,size=(len(I[j]),master_time_horizon+1))
            for i in I[j]:
                for t in T:
                    if(i == 0):
                        need[0][t] += best_product_sum[j][i,t]
                    else:
                        need[1][t] += best_product_sum[j][i,t]
            sum_df = pd.DataFrame(need)
            sum_df.to_excel(writer, sheet_name = "MP_greedy生產加總")
            

