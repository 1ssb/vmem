# Representative 10 Task Table

Selection policy: first representative card per `robot_node` from
`pipeline/benchmark/dataset_card/index.json`.

| # | Job | Side | Object Search | Target Room | Observation | Extension px | Start Crop |
|---:|---|---|---|---|---|---:|---|
| 1 | `01_bed_left_1013` | left | bed | bedroom | partially_in_crop_occluded | 274.48 | `01_bed_left_1013/input_crop.png` |
| 2 | `02_cabinet_left_114` | left | cabinet | kitchen | partially_in_crop_occluded | 122.97 | `02_cabinet_left_114/input_crop.png` |
| 3 | `03_chair_right_227` | right | chair | living_room | partially_in_crop_occluded | 106.54 | `03_chair_right_227/input_crop.png` |
| 4 | `04_microwave_left_236` | left | microwave | kitchen | on_image_outside_crop_occluded | 158.74 | `04_microwave_left_236/input_crop.png` |
| 5 | `05_plant_left_265` | left | plant | living_room | partially_in_crop_occluded | 77.95 | `05_plant_left_265/input_crop.png` |
| 6 | `06_refrigerator_left_170` | left | refrigerator | kitchen | on_image_outside_crop_occluded | 358.30 | `06_refrigerator_left_170/input_crop.png` |
| 7 | `07_shelf_right_567` | right | shelf/bookshelf | living_room | partially_in_crop_occluded | 67.60 | `07_shelf_right_567/input_crop.png` |
| 8 | `08_sink_right_375` | right | sink | bathroom | on_image_outside_crop_occluded | 136.58 | `08_sink_right_375/input_crop.png` |
| 9 | `09_table_left_116` | left | table | living_room | on_image_outside_crop_occluded | 237.18 | `09_table_left_116/input_crop.png` |
| 10 | `10_tv_right_89` | right | tv | bedroom | partially_in_crop_occluded | 161.08 | `10_tv_right_89/input_crop.png` |

## Left Jobs

```text
01_bed_left_1013
02_cabinet_left_114
04_microwave_left_236
05_plant_left_265
06_refrigerator_left_170
09_table_left_116
```

Run:

```bash
python scripts/run_representative_10.py --side left
```

## Right Jobs

```text
03_chair_right_227
07_shelf_right_567
08_sink_right_375
10_tv_right_89
```

Run:

```bash
python scripts/run_representative_10.py --side right
```

## Exact Motion Contract

Every command uses the side stored in `job.json`:

```text
left  -> scripts/generate_simple_video.py --left
right -> scripts/generate_simple_video.py --right
```

Every command also uses:

```text
--camera-json <job>/camera.json
--image-transform-mode pad
--restore-input-size
```

## SegFormer Target Check

Before running the generated videos through the benchmark, verify the segmenter
path is available:

```bash
python scripts/evaluate_representative_10_segformer.py --install-check
```

After VMem outputs exist:

```bash
python scripts/evaluate_representative_10_segformer.py
```

The checker uses the following target class sets:

| Object Search | ADE20K labels |
|---|---|
| bed | bed, cradle |
| cabinet | cabinet, wardrobe, chest of drawers |
| chair | chair, armchair, seat, bench, swivel chair |
| microwave | microwave |
| plant | plant, flower, palm |
| refrigerator | refrigerator |
| shelf | shelf, bookcase, case |
| sink | sink |
| table | table, desk, counter, countertop, coffee table, kitchen island |
| tv | television receiver, monitor, screen, computer, crt screen |
