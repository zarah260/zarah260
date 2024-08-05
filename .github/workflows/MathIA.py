import numpy as np
X = [(50250,5330.85), (41134.1,3174.39), (49755.1,4378.65), (54954.8,6291.04), (31832,2102.77), (63093.5,10528.5),
(20397.3,1116.03), (40720.2,3443.66), (47163.5,4288.65), (43123.6,3522.08), (46569,5136.2), (57462.7,5316.58),
(116644.8,5220.9), (69710.4,6536.8), (34931.8,3152.8), (28884.1,1713.9), (71705.6,6978.1), (53553.3,5456.5),
(40343.1,2836.3), (29680.9,2290.6), (50666.1,5370.44), (43005.6,3417.49), (51556.5,4558.54), (55891.2,6518),
(33956.8,2169.77), (65297.5,10948.48), (20609.1,1132.97), (42184.4,3600.27), (48542.2,4500.14), (44850.9,3653.4),
(49377.1,5274.26), (60334.8,5477.57), (120962.4,5414.5), (68345,6744.6), (36871.9,3347.4), (29400.6,1850.3),
(73114.5,7138.1), (55068.8,5551.9), (41947.6,2903.4), (30869.2,2319)]

Y = (58.67, 33.29, 77.35, 71.53, 31.96, 64.69, 9.25, 31.52, 59.86, 32.73, 58.11, 89.58, 77.18, 106.74, 38.92, 29.98,
82, 84.13, 49.82,24.26, 58.87, 33.99, 80.14, 72.91, 32.6, 66.11, 9.23, 32.41, 60.69, 33.46, 63.9, 90.29, 78.91, 106.96,
     40.23, 29.98, 82.73, 78.51, 50.39, 24.87)

import pandas as pd
df2 = pd.DataFrame(X, columns=['GDP', 'HealthExpenditure'])
df2['EmploymentDensity'] = pd.Series(Y)
df2
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
model = smf.ols(formula='EmploymentDensity ~ GDP + HealthExpenditure ', data=df2)
results_formula = model.fit()

x_surf, y_surf = np.meshgrid(np.linspace(df2.GDP.min(), df2.GDP.max(), 100), np.linspace(df2.HealthExpenditure.min(),
df2.HealthExpenditure.max(), 100))
onlyX = pd.DataFrame({'GDP': x_surf.ravel(), 'HealthExpenditure': y_surf.ravel()})
fittedY = results_formula.predict(exog=onlyX)

fittedY = np.array(fittedY)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df2['GDP'], df2['HealthExpenditure'], df2['EmploymentDensity'], c='red', marker='o', alpha=0.5)
ax.plot_surface(x_surf, y_surf, fittedY.reshape(x_surf.shape), color='b', alpha=0.3)
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
plt.show()
