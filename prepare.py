# PREP

# from bayes methodologies repo, fills zeros in nulls
def fill_zero(df, cols):
    df.fillna(value=0, inplace=True)
    return df

# drop ID column
def drop_id(df):
	return

# make parcelid the index
def set_index_parcel(df):
	return

# renaming columns to accomodate my laziness
def lazy_col_names(df):
	return

# drop unitcnt > 1
def drop_obvious_units(df):
	return

# drop bed = 0
def drop_no_beds(df):
	return

# drop bath = 0
def drop_no_baths(df):
	return

# drop if calculatedfinishedsquarefeet is null
def drop_no_buildings(df):
	return

# drop all condos (imo questionable, at Maggie's behest). condo is type 266
# maybe look at condos.
# drop those with probably >1 unitcnt where unitcnt is null
# unlikely single unit: residential general, duplex, quadruplex, triplex, cluster home, commerical/office/residential mixed use
def drop_probable_units(df):
	bad_types = [31, 246, 247, 248, 260, 265, 266, 263, ]
	return

def prep_zillow(df):
	prep = drop_id(df)
	prep = set_index_parcel(prep)
	prep = lazy_col_names(prep)
	prep = drop_obvious_units(prep)
	prep = drop_no_beds(prep)
	prep = drop_no_baths(prep)
	prep = drop_no_buildings(prep)
	prep = drop_condos(prep)
	prep = drop_probable_units(prep)
	to_zero = []
	prep = fill_zero(prep, to_zero)
	return prep