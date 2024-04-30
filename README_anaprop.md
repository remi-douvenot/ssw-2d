# Projet 2A Anaprop

## Dependencies

In order to load netCDF weather datasets, additionnal depencencies are required :

* ```cdsapi``` : download data from the copernicus datastore
* ```cfgrib``` : access netcdf datasets

These can be installed with the following command.

```pip3 install xarray cfgrib cdsapi```

Along with the eccodes system library (required by cfgrib)

```sudo apt install libeccodes0```

## Setting up API keys

### Slices of terrain with IGN (ok with France)

IGN is a french public geographical organization, it does not require account creation or api key for now.

The data it returns can sometimes be invalid.

### Slices of terrain with Bing (recommended)

In order to load vertical profiles into the software, we ask Bing Maps API for the data, and an API key is required.

Create your API key following [this documentation](https://learn.microsoft.com/en-us/bingmaps/getting-started/bing-maps-dev-center-help/getting-a-bing-maps-key).

Once you have it, set it as an environment variable, in the same shell you'll be running SSW from, by running the command below :

```export BING_MAPS_API_KEY=yourkey```

### Atmospherical datasets with EU Copernicus Datastore (recommended)

Our project accesses the Copernicus ERA dataset, in order for you to access it, create an account [here](https://cds.climate.copernicus.eu/user/register), then set up your api key as explained [here](https://cds.climate.copernicus.eu/api-how-to).

In a nutshell, for linux users, complete the following file in ```$HOME/.cdsapirc``` of the user that will run SSW with the contents below :

```
url: https://cds.climate.copernicus.eu/api/v2
key: yourkey
```

## Anaprop configuration

Copy the example environment file into a working one.

```
cp .env.example .env
```

Then, edit the contents of the ```.env``` file to your needs, the following values can be set (see below).

The variable ```SSW_DATA_PATH``` below **has to be set** if you are using **ERA5** atmosphere.

When you are happy with the values, load the variables into your environment by running ```source .env``` in the shell SSW-2D is run from.

### SSW data path

When SSW downloads datasets from the internet, it stores those in a separate folder (because those binary files can be large). You can set your own directory by setting the variable below. Note that the folder has to exist.

```
export SSW_DATA_PATH=/path/to/big/storage
```

Atmospherical datasets will be downloaded here, terrain datasets are smaller and can be stored in the cache.

### Bing maps API key

Store your Bing API key by adding this to your ```.env```

```
export BING_MAPS_API_KEY=yourkey
```

### SSW cache size

You can change the cache size (number of files) by setting the value below. Default value is 20.

SSW-2D has a rolling cache, it keeps the ```SSW_CACHE_NB``` most recently accessed elements and deletes the old ones.

```
export SSW_CACHE_NB=30
```

### SSW cache path

By default, SSW will store some of it's data into the system's cache folder. (on linux, ```~/.cache/SSW-2D/```). Have a look [here](https://github.com/platformdirs/platformdirs) for more details on each OS cache directory, here ```user_cache_dir()``` is used.

```
export SSW_CACHE_PATH=/path/to/your/cache
```

### Disable SSW caching

When terrain or atmospherical data is requested, SSW-2D will create a hash of it's metadata, then look up in the cache folder for a matching hash.

If a hash is found, the corresponding dataset is directly loaded into SSW-2D.

If the matching hash is not found, the dataset is downloaded and cached in the cache folder using it's hash.

This behavior can be completly disabled by setting the variable below to 1.

```
export SSW_DISABLE_CACHE=1
```

## Into SSW

### Relief tab

Into the relief tab, you can select two points P and Q by clicking "Select Points". A dialog will open.

The point P corresponds to the emitter, and Q corresponds to the receiver.

For each point, a comma-separated latitude and longitude is expected as shown below.

```
34.85545567,4.65435352
```

Note that when P and Q are set, they will also be used to compute the atmosphere slice.

When relief is ran, a terrain profile is downloaded from Bing or IGN using the two points and the sampling N_x.

### Atmosphere tab

In the atmosphere tab, you'll be able to choose ERA5 from the dropdown list. Choose a time for the atmosphere snapshot, note that you'll only be able to select entire hours (for example 14:00). 

For more details on ERA5, please see the [dataset's page](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview).

When the simulation is ran, an atmospherical dataset will be downloaded if it is not found in ```SSW_DATA_PATH```. 

Note that downloading datasets can take a long time, mostly because the data has to be extracted from super large datasets.

When a request is made, it is queued (can take a long time), ```cdsapi``` then checks if the file can be downloaded every second.

Each file is about 2.4MB.