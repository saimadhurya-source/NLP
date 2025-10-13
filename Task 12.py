import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
np.random.seed(42)
data = pd.DataFrame({
'Category': ['A', 'B', 'C', 'D', 'E'],
'Values': np.random.randint(10, 100, 5)
})
plt.figure(figsize=(6, 4))
plt.bar(data['Category'], data['Values'], color='blue')
plt.xlabel('Category')
plt.ylabel('Values')
plt.title('Bar Chart using Matplotlib')
plt.show()
sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
sns.barplot(x='Category', y='Values', data=data, palette='viridis')
plt.title('Bar Chart using Seaborn')
plt.show()
