# Where the location list comes from, and how to improve it

`weather_platform/data/india_cities.csv` has 71 locations: every state capital, every UT capital, and major cities/towns
chosen to cover India's climate zones (Thar desert, Western Ghats coast, Himalayan hills, Northeast, islands).
Coordinates were entered from general knowledge as approximate city-centre points. They are fine for a 0.25 degree
(~25 km) ERA5 grid, but you must VERIFY them:

    python scripts/verify_cities_geonames.py

It downloads GeoNames' India file and reports any location more than 25 km from the matching GeoNames place
(or with no name match, usually a spelling difference such as Bengaluru/Bangalore). Fix the CSV rows it flags.

## Sources for larger lists (check each licence before reuse)
| Source | What you get | Licence (verify on the site) |
|---|---|---|
| GeoNames, `download.geonames.org/export/dump/` (IN.zip, cities15000.zip) | names, lat/lon, population, state code | CC BY 4.0, attribution required |
| Natural Earth "populated places" | major cities, lat/lon | public domain |
| Wikipedia / Wikidata lists of state capitals and districts | names, coordinates via Wikidata | CC BY-SA / CC0 |
| Census of India / data.gov.in district lists | official names, usually no coordinates | open government data terms |

To extend the list: add rows to the CSV (name,state,kind,region,lat,lon). Keep points at least ~0.15 degrees apart
(tests/test_cities.py enforces this) because points inside one ERA5 grid cell give identical data.
Region labels are a simplified grouping (North, South, East, West, Central, Northeast, Islands) used for display only.
Haryana has no row of its own: Chandigarh is its (shared) capital; add Gurugram or Hisar if you want it represented.
