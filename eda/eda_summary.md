# Pancrea Cyst Dataset EDA

- Data directory: `/home/huy/quan_nguyen/aima/pancrea_cyst/data`
- Image files: 358
- Mask files: 358
- Paired cases by ID: 358
- Images without masks: 0
- Masks without images: 0
- Empty positive masks: 0
- Shape mismatches: 0
- Affine mismatches: 17
- Observed mask values: {'0': 358, '1': 341, '1.0000000591389835': 17}

## Splits

| split     |   rows |   unique_cases |   mismatched_image_mask_ids |   missing_image_files |   missing_mask_files |
|:----------|-------:|---------------:|----------------------------:|----------------------:|---------------------:|
| train     |    247 |            247 |                           0 |                     0 |                    0 |
| val       |     37 |             37 |                           0 |                     0 |                    0 |
| test      |     74 |             74 |                           0 |                     0 |                    0 |
| all_train |    284 |            284 |                           0 |                     0 |                    0 |

| site   |   all_train |   test |   train |   val |
|:-------|------------:|-------:|--------:|------:|
| AHN    |          11 |      3 |       9 |     2 |
| CAD    |          66 |     17 |      58 |     8 |
| EMC    |          28 |      7 |      24 |     4 |
| IU     |          28 |      7 |      24 |     4 |
| MCA    |          12 |      3 |      10 |     2 |
| MCF    |          15 |      4 |      13 |     2 |
| NU     |          68 |     18 |      60 |     8 |
| NYU    |          56 |     15 |      49 |     7 |

## Geometry

- Unique image shapes: 180
- Unique voxel spacings: 195
- Z spacing stats (mm): {'min': 2.9999961853027344, 'p25': 4.400001525878906, 'median': 5.0, 'mean': 5.607749790452712, 'p75': 6.599999904632568, 'max': 15.857142448425293}
- Voxel volume stats (mm^3): {'min': 1.953125, 'p25': 4.273942717251016, 'median': 6.042472741683014, 'mean': 6.817965959843344, 'p75': 8.43463134765625, 'max': 23.125810591811046}

## Mask Volumes

- Mask volume stats (mL): {'min': 0.084609375, 'p25': 2.764696485324646, 'median': 8.066224121481355, 'mean': 18.515564004526908, 'p75': 20.893397248722252, 'max': 257.107431691722}
- Median mask fraction of image volume: 0.00034854

### 10 Largest Masks

| case_id   | site   |   shape_x |   shape_y |   shape_z |   spacing_x_mm |   spacing_y_mm |   spacing_z_mm |   mask_volume_ml |
|:----------|:-------|----------:|----------:|----------:|---------------:|---------------:|---------------:|-----------------:|
| CAD242    | CAD    |       256 |       256 |        48 |       1.09375  |       1.02539  |        4.4     |          257.107 |
| NYU0035   | NYU    |       320 |       240 |        45 |       1.1875   |       1.1875   |        4.8     |          171.107 |
| MCF20     | MCF    |       320 |       320 |        36 |       0.90625  |       0.90625  |        4.4     |          167.653 |
| MCF27     | MCF    |       320 |       320 |        40 |       0.90625  |       0.90625  |        4.4     |          165.044 |
| MCF25     | MCF    |       512 |       512 |        32 |       0.7813   |       0.7813   |        5.3     |          144.737 |
| MCA15     | MCA    |       512 |       512 |        56 |       0.78125  |       0.78125  |        6       |          134.634 |
| EMC091    | EMC    |       512 |       512 |        31 |       0.7813   |       0.7813   |        6.59998 |          127.174 |
| NYU0130   | NYU    |       320 |       320 |        50 |       1.40625  |       1.40625  |        6       |          126.104 |
| EMC009    | EMC    |       512 |       512 |        34 |       0.7813   |       0.7813   |        6.60001 |          124.757 |
| EMC097    | EMC    |       528 |       528 |        25 |       0.710227 |       0.710227 |        8       |          118.911 |

### 10 Smallest Non-empty Masks

| case_id   | site   |   shape_x |   shape_y |   shape_z |   spacing_x_mm |   spacing_y_mm |   spacing_z_mm |   mask_volume_ml |
|:----------|:-------|----------:|----------:|----------:|---------------:|---------------:|---------------:|-----------------:|
| NU63      | NU     |       320 |       250 |        44 |        1.1875  |        1.1875  |        6       |        0.0846094 |
| NYU0076   | NYU    |       320 |       270 |        40 |        1.09375 |        1.09375 |        6       |        0.114844  |
| CAD137    | CAD    |       256 |       256 |        30 |        0.9375  |        0.9375  |        4.4     |        0.12375   |
| NYU0022   | NYU    |       320 |       320 |        34 |        1.1875  |        1.1875  |        5       |        0.169219  |
| NU144     | NU     |       320 |       310 |        57 |        1.125   |        1.125   |        5       |        0.423984  |
| NU145     | NU     |       256 |       208 |        55 |        1.5625  |        1.5625  |        5       |        0.45166   |
| NYU0023   | NYU    |       320 |       220 |        40 |        1.09375 |        1.09375 |        6.00001 |        0.516797  |
| NU184     | NU     |       320 |       296 |        54 |        1.1875  |        1.1875  |        5       |        0.528809  |
| CAD167    | CAD    |       256 |       256 |        24 |        1.01562 |        1.01562 |        4.4     |        0.54009   |
| NU127     | NU     |       320 |       260 |        40 |        1.25    |        1.25    |        6       |        0.54375   |

## Output Files

- `case_level_eda.csv`
- `cases_by_site.png`
- `mask_volume_hist.png`
- `mask_volume_log_hist.png`
- `representative_mask_overlay.png`
- `site_by_split.csv`
- `slice_count_vs_spacing.png`
- `split_summary.csv`
