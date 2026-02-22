from scipy.stats import ttest_ind

def t_test(a, b):
    stat, p = ttest_ind(a, b)
    return p