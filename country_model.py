# -*- coding: utf-8 -*-
import numpy as np
import pymc as pm
import arviz as az
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import pytensor.tensor as pt

np.random.seed(27)


#####plotting parameters
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.titlesize': 12})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"

df = pd.read_csv("./data/Dengue_Occurrence_12122013_Points.csv")
df = df.dropna()

pmap = gpd.read_file("./data/AllDengue.shp")

# wmap = gpd.read_file("./data/WB_countries_Admin0_10m/WB_countries_Admin0_10m.shp")
wmap = gpd.read_file("./data/wmap/wmap.shp")  
wmap = wmap[wmap.NAME_ENGLI != "Antarctica"]

d1 = {'Viet Nam': 'Vietnam',
 'United Republic of Tanzania': "Tanzania",
 'Turks and Caicos islands': 'Turks and Caicos Islands',
 "Lao People's Democratic Republic": 'Laos',
 'United States of America':'United States',
 'United States Virgin Islands': 'Virgin Islands, U.S.',
 'Micronesia (Federated States of)':'Micronesia',
 "Cote d'Ivoire":"Côte d'Ivoire",
 'Jammu Kashmir': "India",
 "Netherlands Antilles": 'Curaçao'}
 
d2  = {"People's Republic of China": "China",
'Federated States of Micronesia': 'Micronesia',
'East Timor': 'Timor-Leste',
'The Bahamas': 'Bahamas'}

df.COUNTRY = df.COUNTRY.replace(d1)
pmap.COUNTRY = pmap.COUNTRY.replace(d1)
wmap.NAME_EN = wmap.NAME_ENGLI.replace(d2)

co = [c for c in wmap.NAME_EN.unique() if c in df.COUNTRY.unique()]
no_co = [c for c in df.COUNTRY.unique() if c not in co]

df = df[~df.COUNTRY.isin(no_co)]
pmap = pmap[~pmap.COUNTRY.isin(no_co)]

df['outbreaks'] = np.repeat(1, len(df)) #outbreak rate 
df_ave = df[["COUNTRY", "YEAR", "outbreaks"]]
df_ave = df_ave.groupby(["COUNTRY", "YEAR"], as_index=False).sum()
df_ave = df_ave.groupby("COUNTRY", as_index=False).mean()

mapcolor = "silver"
color = "steelblue"

wmap["COUNTRY"] = wmap.NAME_EN
wmap2 = pd.merge(wmap, df_ave)
wmap2['log_outbreaks'] = np.log(wmap2.outbreaks)

max_val = wmap2.log_outbreaks.values.max().round(0) 

### plot by cpountry
fig, ax = plt.subplots(figsize=(12,10))
im = ax.imshow(np.arange(4, 0, -1).reshape(2, 2)/4, cmap='plasma', origin="lower", vmin=0, vmax=max_val)
im.set_visible(False)
wmap.plot(color=mapcolor, edgecolor="k", linewidth=0.1, ax=ax)
wmap2.plot(column="log_outbreaks", cmap='plasma', ax=ax, linewidth=0)
ax.axis("off")
cbar = fig.colorbar(im, orientation='horizontal', shrink=0.4, pad=0.01)
cbar.ax.tick_params(labelsize=12) 
cbar.set_label(label="Log Mean Outbreaks")
plt.title("Average Dengue Outbreaks (1960-2012)", size=18)
plt.tight_layout()
plt.savefig("average_dengue_outbreaks.png", dpi=600, bbox_inches='tight')
plt.close()


### plot by year
ntraj=52
cmap = plt.cm.jet(np.linspace(0,1,ntraj))# Initialize holder for trajectories
fig, ax = plt.subplots(figsize=(12,10))
im = ax.imshow(np.arange(4, 0, -1).reshape(2, 2)/4, cmap='jet', origin="lower", vmin=1960, vmax=2012)
im.set_visible(False)
wmap.plot(color=mapcolor, edgecolor="k", linewidth=0.1, ax=ax)
c = 0
for y in df.YEAR.unique():
    ax.scatter(df[df.YEAR==y].X, df[df.YEAR==y].Y, color=cmap[c], marker=",", s=1)
    c = c+1
ax.axis("off")
cbar = fig.colorbar(im, orientation='horizontal', shrink=0.4, pad=0.01)
cbar.ax.tick_params(labelsize=12) 
cbar.set_label(label="Year", size=12)
plt.title("Dengue Outbreaks (1960-2012)", size=18)
plt.tight_layout()
plt.savefig("yearly_dengue_outbreaks.png", dpi=600, bbox_inches='tight')
plt.close()


df['coords'] = [(df.X.values[i], df.Y.values[i]) for i in range(len(df))]


wmap3 = pd.merge(wmap, df_ave, how="left")
wmap3['outbreaks'] = np.nan_to_num(wmap3['outbreaks'].values, nan=0)

coords = wmap3['geometry'].apply(lambda x: x.representative_point().coords[:])

xy = np.array([c[0] for c in coords])

x_data = xy

y_data = wmap3.outbreaks.values

coords = {"country":wmap3.COUNTRY.unique(), 
          "feature":["longitude", "latitude"]}

with pm.Model(coords=coords) as mod:
    X = pm.ConstantData("X", x_data, dims=("country", "feature"))
    s = pm.HalfNormal("s", 1)
    k = pm.gp.cov.ExpQuad(input_dim=2, ls=s)
    latent = pm.gp.Latent(cov_func=k,)
    f = latent.prior("f", X, dims="country")
    a = pm.Beta("a", 1, 1)
    y = pm.ZeroInflatedPoisson('y', mu=pm.math.exp(f), psi=a, observed=y_data)

    
with mod:
    idata = pm.sample(1000, tune=1000, chains=4, cores=12, nuts_sampler="numpyro", target_accept=0.9)

# az.to_netcdf(idata, "country_model_idata.nc")


f_pos = az.extract(idata.posterior['f'])['f'].values


wmap3['intensity_mean'] = f_pos.mean(axis=1) #/ f_pos.mean(axis=1).max() #normalized intensity
wmap3['intensity_sd'] = f_pos.std(axis=1) #/ f_pos.std(axis=1).max() #normalized intensity

max_val = wmap3.intensity_mean.values.max().round(0) -1
min_val = wmap3.intensity_mean.values.min().round(0) 
### plot mean intensity
fig, ax = plt.subplots(figsize=(12,10))
im = ax.imshow(np.arange(4, 0, -1).reshape(2, 2)/4, cmap='coolwarm', origin="lower", vmin=-max_val, vmax=max_val)
im.set_visible(False)
wmap.plot(color=mapcolor, edgecolor="k", linewidth=0.1, ax=ax)
wmap3.plot(column="intensity_mean", cmap='coolwarm', ax=ax, linewidth=0, vmin=-max_val, vmax=max_val)
ax.axis("off")
cbar = fig.colorbar(im, orientation='horizontal', shrink=0.4, pad=0.01)
cbar.ax.tick_params(labelsize=12) 
cbar.set_label(label="Intensity (mean)", size=12)
plt.title("Outbreaks Intensity Posterior Means (1960-2012)", size=18)
plt.tight_layout()
plt.savefig("mean_estimate_dengue_outbreaks_country_model.png", dpi=600, bbox_inches='tight')
plt.close()

max_val = wmap3.intensity_sd.values.max().round(0) 
min_val = wmap3.intensity_sd.values.min().round(0) 
### plot sd intensity
fig, ax = plt.subplots(figsize=(12,10))
im = ax.imshow(np.arange(4, 0, -1).reshape(2, 2)/4, cmap='Reds', origin="lower", vmin=min_val, vmax=max_val)
im.set_visible(False)
wmap.plot(color=mapcolor, edgecolor="k", linewidth=0.1, ax=ax)
wmap3.plot(column="intensity_sd", cmap='Reds', ax=ax, linewidth=0)
ax.axis("off")
cbar = fig.colorbar(im, orientation='horizontal', shrink=0.4, pad=0.01)
cbar.ax.tick_params(labelsize=12) 
cbar.set_label(label="Intensity (SD)", size=12)
plt.title("Outbreaks Intensity Posterior standard deviations (1960-2012)", size=18)
plt.tight_layout()
plt.savefig("sd_estimate_dengue_outbreaks_country_model.png", dpi=600, bbox_inches='tight')
plt.close()


## save idata summary
summ = az.summary(idata, hdi_prob=0.9)
summ.to_csv("country_model_idata_summary.csv")


### Plot traces as rankplots
az.plot_trace(idata, var_names=["s", "a"], kind="rank_vlines")
plt.rcParams['axes.titlesize'] = 30
plt.suptitle("Model 1 (Country-level)", size=18)
plt.tight_layout()
plt.savefig("rankplots_country_model.png", dpi=600)