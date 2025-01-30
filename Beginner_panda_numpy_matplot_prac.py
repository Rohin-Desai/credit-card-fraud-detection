import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt




months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
stores = ["Store A", "Store B", "Store C", "Store D", "Store E"]

np.random.seed(42)

sales_data = np.random.randint(5000, 20000, size =(5,6))

df = pd.DataFrame(sales_data, index=stores, columns = months)



mean_per_store = df.mean(axis = 1)
print(mean_per_store)

max , min = sales_data.max(), sales_data.min()

print(max,min)

sum = df.sum(axis = 1)

print(sum)

best = sum.idxmax(axis = 0)
worse = sum.idxmin(axis = 0)

print(best)
print(worse)

months1 = df.columns

store_sales = {}

for store, sales in df.iterrows():
    store_sales[store] = sales.values 

for store in df.index:
    x = months
    y = store_sales[store]
    
    plt.plot(x,y, label = store)
    plt.grid()
    plt.xlabel("month")
    plt.ylabel("sales price")
    plt.legend()
    

plt.show()

total_sales_array = df.sum(axis = 1)

print(total_sales_array)

for i, store in enumerate(total_sales_array.index):
    x = store
    y = total_sales_array.iloc[i]
    plt.bar(x,y, color = 'red')
    plt.xlabel("Store name")
    plt.ylabel("sales price")


plt.show()
values = []
categories = []
for i, store in enumerate(total_sales_array.index):
    values.append(total_sales_array.iloc[i])
    categories.append(store)

plt.pie(values, labels = categories, autopct = '%1.1f%%')

plt.show()

max_per_store = df.idxmax(axis = 1)

print(max_per_store)





















