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

### IGN

IGN is a french public geographical organization, it does not require account creation or api key for now.

The data it returns can sometimes be invalid.

### Bing (recommended)

In order to load vertical profiles into the software, we ask Bing Maps API for the data, and an API key is required.

Create your API key following [this documentation](https://learn.microsoft.com/en-us/bingmaps/getting-started/bing-maps-dev-center-help/getting-a-bing-maps-key).

Once you have it, set it as an environment variable, in the same shell you'll be running SSW from, by running the command below :

```export BING_MAPS_API_KEY=yourkey```

### EU Copernicus Datastore (recommended)

Our project accesses the Copernicus ERA dataset, in order for you to access it, create an account [here](https://cds.climate.copernicus.eu/user/register), then set up your api key as explained [here](https://cds.climate.copernicus.eu/api-how-to).

In a nutshell, for linux users, complete the following file in ```$HOME/.cdsapirc``` of the user that will run SSW with the contents below :

```
url: https://cds.climate.copernicus.eu/api/v2
key: yourkey
```

## Into SSW