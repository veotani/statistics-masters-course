import numpy as np
from scipy.stats import norm

# Входные параметры
a = len('Орлов')
b = len('Антон')
n = 100

# Проверка критерия Колмогорова-Смирнова по выборке и её функции распределения (теор.)
def kolmogorov_smirnov(X, F_func, D_crit):

    n = len(X)

    # Построим функцию F и Fn для поиска статистики Dn
    #F = np.array([1-np.exp(-la*x) for x in np.arange(min(X), max(X), 0.001)])
    F = np.array([F_func(x) for x in np.arange(min(X), max(X), 0.001)])
    Fn = np.array([len(X[np.where(X<x)])/n for x in np.arange(min(X), max(X), 0.001)])
    Dn = max(np.abs(F-Fn))

    # Критическое значение:
    D_crit = 1.36/np.sqrt(n)       # Множество таблиц ссылаются на эту формулу при уровне значимости alpha = 0.95
                                   # Она используется только при значениях n > 50

    # Делаем вывод
    if Dn < D_crit:
        print('Dn < D_crit => гипотеза H_0 принимается и введённая выборка принадлежит введённому распределению')
        return True
    else:
        print('Dn >= D_crit => гипотеза H_0 отвергается в пользу альтернативной: выборка не принадлежит изучаемому распределению')
        return False

# Задание 1, 2: генерация выборки и проверка гипотезы
# Считаем параметры распределения и строим его
la = a/b
betta = 1/la
F_expon = lambda x: 1-np.exp(-la*x)
X_expon = np.random.exponential(1/la, n)
# Множество таблиц ссылаются на эту формулу при уровне значимости alpha = 0.95
# Она используется только при значениях n > 50
D_crit = 1.36/np.sqrt(len(X_expon))
print('Для экспоненциального распеделения:')
kolmogorov_smirnov(X_expon, F_expon, D_crit)
# Dn < D_crit => гипотеза H_0 принимается и введённая выборка принадлежит введённому распределению

#Задание 3

# Для начала немного теории: для равномерного распределения параметрами являются не мат.ожидание и дисперсия, а a и b
# Их можно найти из следующей системы:
# (a+b)/2 = m
# (a-b)^2/12 = d
a = 5-np.sqrt(15)
b = 5+np.sqrt(15)
n = 40
X_ravnom = []
for i in range(15):
    X_ravnom.append(np.random.uniform(a, b, 40))
#F_ravnom = lambda x: (x-a)/(b-a)
F_norm = lambda x: norm.cdf(x, 5, 5)
res_ravnom = []
for x in X_ravnom:
    res_ravnom.append(kolmogorov_smirnov(x, F_norm, 0.21017))
res_ravnom = np.array(res_ravnom)
print('Для 15 выборок из равномерного распределения:')
successes = len(res_ravnom[np.where(res_ravnom == True)])
fails = len(res_ravnom[np.where(res_ravnom == False)])
print('{} принятых H0 и {} отвергнутых в пользу H1'.format(successes, fails))
# Вывод: для этих выборок отвергается нулевая гипотеза в пользу альтернативной -- они не принадлежат нормальным распределениям с мат.ождианием 5 и дисперсией 5

# Задание 4
X_ravnom_means = np.array([x.mean() for x in X_ravnom])
F_norm_4 = lambda x: norm.cdf(x, 5, 5/40)
print('Для совокупности средних выборок из прошлого задания:')
kolmogorov_smirnov(X_ravnom_means, F_norm_4, 0.33760)
# Вывод: нулевая гипотеза отвергается в пользу альтернативной -- выборка не принадлежит нормальному распределению с мат.ожиданием 5 и дисперсией 1/8

# В последних заданиях мы получили опровержение нулевых гипотез, что не является удивительным, поскольку наши выборки принадлежат другому распределению и тест Колмогорова-Смирнова помог это выявить
