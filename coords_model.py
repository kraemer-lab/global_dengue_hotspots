# -*- coding: utf-8 -*-
import numpy as np
import pymc as pm
import arviz as az
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import pytensor.tensor as pt
from tqdm import tqdm
import seaborn as sns
from itertools import product

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


xy = df[['X', 'Y']].values

eps = 1e-3
rng = np.random.default_rng()
xy = xy.astype("float") + rng.standard_normal(xy.shape) * eps

coords = wmap['geometry'].apply(lambda x: x.representative_point().coords[:])

xy2 = np.array([c[0] for c in coords])

resolution = 40

area_per_cell = resolution ** 2 / 100

cells_x = 100
cells_y = 40

# Creating bin edges for a 2D histogram
quadrant_x = np.linspace(np.min(xy2[:, 0])-10, np.max(xy2[:, 0])+10, cells_x + 1)
quadrant_y = np.linspace(np.min(xy2[:, 1])-10, np.max(xy2[:, 1])+10, cells_y + 1)

# Identifying the midpoints of each grid cell
centroids = np.asarray(list(product(quadrant_x[:-1], quadrant_y[:-1])))

cell_counts, edge_x, edge_y = np.histogram2d(xy[:, 0], xy[:, 1], [quadrant_x, quadrant_y])
cell_counts = cell_counts.ravel().astype(int)

cent_x = edge_x[:-1] + (edge_x[1] - edge_x[0]) / 2
cent_y = edge_y[:-1] + (edge_y[1] - edge_y[0]) / 2

### plot by year gridded
line_kwargs = {"color": "k", "linewidth": 1, "alpha": 0.5}
ntraj=52
cmap = plt.cm.jet(np.linspace(0,1,ntraj))# Initialize holder for trajectories
fig, ax = plt.subplots(figsize=(12,10))
im = ax.imshow(np.arange(4, 0, -1).reshape(2, 2)/4, cmap='jet', origin="lower", vmin=1960, vmax=2012)
im.set_visible(False)
wmap.plot(color=mapcolor, edgecolor="k", linewidth=0.1, ax=ax)
[ax.axhline(y, **line_kwargs) for y in quadrant_y]
[ax.axvline(x, **line_kwargs) for x in quadrant_x]
c = 0
for y in df.YEAR.unique():
    ax.scatter(df[df.YEAR==y].X, df[df.YEAR==y].Y, color=cmap[c], marker=",", s=1, alpha=0.3)
    c = c+1
for i in range(len(centroids)):
    plt.annotate(cell_counts[i], centroids[i], alpha=0.75, fontsize=5)
ax.axis("off")
cbar = fig.colorbar(im, orientation='horizontal', shrink=0.4, pad=0.01)
cbar.ax.tick_params(labelsize=12) 
cbar.set_label(label="Year", size=12)
plt.title("Dengue Outbreaks (1960-2012) counts per grid cell", size=18)
plt.tight_layout()
plt.savefig("grided_yearly_dengue_outbreaks.png", dpi=600, bbox_inches='tight')
plt.close()

X = [cent_x[:, None], cent_y[:, None]]
Y = cell_counts

####### Model ##########
with pm.Model() as mod:
    sm = pm.HalfNormal('sm',  1) #length scale lon
    sn = pm.HalfNormal('sn',  1) #length scale lat
    Km = pm.gp.cov.ExpQuad(1, ls=sm) #covariance function x coordinates
    Kn = pm.gp.cov.ExpQuad(1, ls=sn) #covariance function y coordinates
    gp = pm.gp.LatentKron(cov_funcs=[Km, Kn])
    lam = gp.prior('lam', Xs=X)
    a = pm.HalfNormal("a", 1)
    y = pm.NegativeBinomial('y', mu=pm.math.exp(lam), alpha=a, observed=Y)

with mod:
    idata = pm.sample(2000, tune=2000, chains=4, cores=12, nuts_sampler="numpyro", target_accept=0.95)
    
int_pos = az.extract(idata.posterior['lam'])['lam'].values #intensity posterior
pos_mean = int_pos.mean(axis=1)
posm_rate = np.exp(int_pos).mean(axis=1)
pos_sd = int_pos.std(axis=1)
pmin = pos_mean.min()
pmax = pos_mean.max()

pos_df = pd.DataFrame({"pos_mean":pos_mean, "pos_sd":pos_sd, 
                       "posm_rate":posm_rate*area_per_cell,
                       "x":centroids[:, 0], "y":centroids[:, 1]})

bins =  (quadrant_x, quadrant_y)

max_val = 2 #pos_mean.max()
min_val = -2 #pos_mean.min()

### plot intensity heatmap means
fig, ax = plt.subplots(figsize=(12,10))
im = ax.imshow(np.arange(4, 0, -1).reshape(2, 2)/4, cmap='coolwarm', alpha=0.5, origin="lower", vmin=min_val, vmax=max_val)
im.set_visible(False)
wmap.plot(facecolor="w", edgecolor="k", linewidth=1, ax=ax)
ax.scatter(df.X, df.Y, color="#069476", marker="o", s=5, alpha=0.3, label="Outbreaks")
ax.legend(handletextpad=0.1, markerscale=3)
ax.hist2d(x="x", y="y", weights="pos_mean", vmin=min_val, vmax=max_val, cmap="coolwarm", alpha=0.5, bins=bins, data=pos_df)
ax.axis("off")
cbar = fig.colorbar(im, orientation='horizontal', shrink=0.4, pad=0.01)
cbar.ax.tick_params(labelsize=12) 
cbar.set_label(label="Intensity", size=12)
plt.title("Dengue Outbreaks Intensity mean (1960-2012)", size=18)
plt.tight_layout()
plt.savefig("intensity_grided_dengue_outbreaks_coords_model.png", dpi=600, bbox_inches='tight')
plt.close()  



### truncate cmap for custom scale (from stackoverflow)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

arr = np.linspace(0, 50, 100).reshape((10, 10))
fig, ax = plt.subplots(ncols=2)

cmap = plt.get_cmap('gist_heat_r')
new_cmap = truncate_colormap(cmap, 0.0, 0.7)
ax[0].imshow(arr, interpolation='nearest', cmap=cmap)
ax[1].imshow(arr, interpolation='nearest', cmap=new_cmap)
plt.show()

### plot intensity heatmap 2
min_val=0
max_val = 250 #pos_df.posm_rate.values.max().round(0)
fig, ax = plt.subplots(figsize=(12,10))
im = ax.imshow(np.arange(4, 0, -1).reshape(2, 2)/4, cmap=new_cmap, alpha=0.8, origin="lower", vmin=min_val, vmax=max_val)
im.set_visible(False)
wmap.plot(facecolor="w", edgecolor="k", alpha=1, linewidth=1, ax=ax)
ax.scatter(df.X, df.Y, color="#16B392", marker="o", s=5, alpha=0.1)
ax.scatter((0,1),(0,1), color="#16B392", marker="o", s=0.01, alpha=0.8, label="Outbreaks")
ax.legend(handletextpad=0.1, markerscale=70, loc="best")
ax.hist2d(x="x", y="y", weights="posm_rate", vmin=min_val, vmax=max_val, cmap=new_cmap, alpha=0.5, bins=bins, data=pos_df)
ax.axis("off")
cbar = fig.colorbar(im, orientation='horizontal', shrink=0.4, pad=0.01)
cbar.ax.tick_params(labelsize=12) 
cbar.set_label(label="Rate", size=12)
plt.title("Dengue Outbreaks Rate mean (1960-2012)", size=18)
plt.tight_layout()
plt.savefig("rate_grided_dengue_outbreaks_coords_model.png", dpi=600, bbox_inches='tight')
plt.close() 



### plot intensity heatmap SDs
arr = np.linspace(0, 50, 100).reshape((10, 10))
fig, ax = plt.subplots(ncols=2)
cmap = plt.get_cmap('Purples_r')
new_cmap = truncate_colormap(cmap, 0.0, 0.8)
ax[0].imshow(arr, interpolation='nearest', cmap=cmap)
ax[1].imshow(arr, interpolation='nearest', cmap=new_cmap)
plt.show()

fig, ax = plt.subplots(figsize=(12,10))
im = ax.imshow(np.arange(4, 0, -1).reshape(2, 2)/4, cmap="bone", alpha=0.8, origin="lower", vmin=0, vmax=1)
im.set_visible(False)
wmap.plot(facecolor="w", edgecolor="k", linewidth=1, ax=ax)
ax.scatter(df.X, df.Y, color="#069476", marker="o", s=5, alpha=0.3, label="Outbreaks")
ax.legend(handletextpad=0.1, markerscale=3)
ax.hist2d(x="x", y="y", weights="pos_sd", vmin=0, vmax=1, cmap="bone", alpha=0.8, bins=bins, data=pos_df)
ax.axis("off")
cbar = fig.colorbar(im, orientation='horizontal', shrink=0.4, pad=0.01)
cbar.ax.tick_params(labelsize=12) 
cbar.set_label(label="Intensity", size=12)
plt.title("Dengue Outbreaks Intensity SD (1960-2012)", size=18)
plt.tight_layout()
plt.savefig("sd_intensity_grided_dengue_outbreaks_coords_model.png", dpi=600, bbox_inches='tight')
plt.close()  



#save idata summary
summ = az.summary(idata, hdi_prob=0.9)
summ.to_csv("coord_model_idata_summary.csv")


### Plot traces as rankplots
az.plot_trace(idata, var_names=["sm", "sn"], kind="rank_vlines")
plt.rcParams['axes.titlesize'] = 30
plt.suptitle("Model 2 (coordinate-level)", size=18)
plt.tight_layout()
plt.savefig("rankplots_coords_model.png", dpi=600)