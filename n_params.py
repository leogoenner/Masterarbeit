import pandas as pd

def n_params_standard(n):
    return n[0]*n[1]*n[2]*n[3]

def n_params_tucker_2(n,r):
    return r[0]*r[1]*r[2]*r[3] + n[2]*r[2] + n[3]*r[3]

data = {}

# SimpleMNIST
n = (3,3,32,64)
r = (3,3,12,16)
data["SimpleMNIST"] = [n, r[2:], n_params_standard(n), n_params_tucker_2(n,r), n_params_tucker_2(n,r)/n_params_standard(n)]

# CustomCIFAR10
n = (3,3,256,256)
r = (3,3,32,32)
data["CustomCIFAR10"] = [n, r[2:], n_params_standard(n), n_params_tucker_2(n,r), n_params_tucker_2(n,r)/n_params_standard(n)]

# VGG16
n = (3,3,512,512)
r = (3,3,40,40)
data["VGG16"] = [n, r[2:], n_params_standard(n), n_params_tucker_2(n,r), n_params_tucker_2(n,r)/n_params_standard(n)]

df = pd.DataFrame(data, index=["Kernel size", "Ranks", "Standard model", "Tucker-2-model", "Ratio"])

# Formatting based on https://stackoverflow.com/a/15070110
def f(x):
    if isinstance(x,int):
        return f'{x:,}'
    elif isinstance(x,float):
        return f'{x:,.2f}'
    else:
        return x
df.to_latex("Masterarbeit/fig/n_parameters.tex", formatters=[f,f,f])

print(df)