# CV-Solar-Irradiance-Version-1

This repository provides a deep learning pipeline for solar irradiance forecasting using both sky images and historical time-series data. The goal is to predict future solar irradiance values, such as global horizontal irradiance (GHI), by leveraging visual and numerical information from past observations.

The project integrates multiple components:

- A convolutional neural network (CNN) to extract features from sequences of sky images.
- A time-series encoder to process historical numerical data such as irradiance measurements.
- A cross-modal fusion module that combines image and time-series embeddings for improved forecasting.
- Temporal modeling using causal convolutions and a transformer encoder to capture sequential patterns.
- A regression head that outputs multi-step forecasts over a defined horizon.

This pipeline is designed to be modular, allowing users to:

- Train a model on custom datasets.
- Validate and evaluate performance with detailed metrics.
- Generate visualizations of forecast errors and predicted versus actual values.

The repository includes scripts for dataset handling, model definition, training, validation, and plotting results. Together, these scripts form a complete workflow for solar irradiance forecasting.

---

## How to Run the Scripts

Follow the steps below to train the model, evaluate it, and generate visualizations of forecasting results.

### 1. Training the Model

The training script prepares the datasets, trains the multimodal forecasting model, and saves checkpoints and the best model.

```bash
python train.py
```
**What happens:**

#### Dataset

The dataset is a CSV file with the following structure:

| timestamp       | raw_image_path                                             | ghi  | dni  | dhi  |
|-----------------|-----------------------------------------------------------|------|------|------|
| 9/1/2019 6:42   | datasets/cropped_raw_images/09_Metas/01/06/20190901064200_00160.jpg | 3.8  | 0    | 3.5  |
| 9/1/2019 6:43   | datasets/cropped_raw_images/09_Metas/01/06/20190901064300_00160.jpg | 4.5  | 0    | 4.3  |
| 9/1/2019 6:44   | datasets/cropped_raw_images/09_Metas/01/06/20190901064400_00160.jpg | 5.3  | 0    | 5.1  |
| 9/1/2019 6:45   | datasets/cropped_raw_images/09_Metas/01/06/20190901064500_00160.jpg | 6.1  | 0    | 5.9  |
| 9/1/2019 6:46   | datasets/cropped_raw_images/09_Metas/01/06/20190901064600_00160.jpg | 6.8  | 0    | 6.6  |
| 9/1/2019 6:47   | datasets/cropped_raw_images/09_Metas/01/06/20190901064700_00160.jpg | 7.6  | 0    | 7.4  |
| 9/1/2019 6:48   | datasets/cropped_raw_images/09_Metas/01/06/20190901064800_00160.jpg | 8.4  | 0    | 8.2  |
| 9/1/2019 6:49   | datasets/cropped_raw_images/09_Metas/01/06/20190901064900_00160.jpg | 9.3  | 0    | 9.1  |
| 9/1/2019 6:50   | datasets/cropped_raw_images/09_Metas/01/06/20190901065000_00160.jpg | 10.1 | 0    | 10   |
| 9/1/2019 6:51   | datasets/cropped_raw_images/09_Metas/01/06/20190901065100_00160.jpg | 11   | 1.3  | 10.9 |
| 9/1/2019 6:52   | datasets/cropped_raw_images/09_Metas/01/06/20190901065200_00160.jpg | 11.9 | 19   | 11.8 |
| 9/1/2019 6:53   | datasets/cropped_raw_images/09_Metas/01/06/20190901065300_00160.jpg | 13.2 | 40   | 12.7 |

**Columns explained:**

- `timestamp`: Date and time of the sample.  
- `raw_image_path`: Path to the corresponding sky image.  
- `ghi`: Global Horizontal Irradiance.  
- `dni`: Direct Normal Irradiance.  
- `dhi`: Diffuse Horizontal Irradiance.  

---

#### Training Script Overview

**Main steps in `train.py`:**

1. **Dataset preparation**:  
   - `IrradianceForecastDataset` extracts sequences of images (`IMG_SEQ_LEN`) and time-series features (`TS_SEQ_LEN`).  
   - Training and validation splits are created.  

2. **Normalization**:  
   - Computes mean and standard deviation for time-series features from the training set.  
   - Saves normalization stats to `norm_stats.json` for reproducible preprocessing.

3. **Model initialization**:  
   - `ImageEncoder` extracts features from sky images.  
   - `MultimodalForecaster` fuses image features with time-series embeddings to predict irradiance values over a forecast horizon (`HORIZON`).  

4. **Training loop**:  
   - Uses mixed-precision training (`torch.cuda.amp`) for efficiency.  
   - Loss is computed using `MSELoss` and optimized with `AdamW`.  
   - Tracks training and validation losses per epoch.  
   - Saves best model weights to `best_model.pth`.  
   - Checkpoints the training state in `checkpoint.pth` for resuming.  

5. **Visualization**:  
   - Saves a loss curve to `training_curve.png`.

---

#### Adjustable Parameters

| Parameter      | Description |
|----------------|-------------|
| `CSV_PATH`     | Path to your dataset CSV file. |
| `BATCH_SIZE`   | Number of samples per batch. |
| `NUM_EPOCHS`   | Total number of epochs. |
| `IMG_SEQ_LEN`  | Number of past sky images per sample. |
| `TS_SEQ_LEN`   | Number of past time-series steps per sample. |
| `HORIZON`      | Forecast horizon (number of future steps to predict). |
| `TARGET_DIM`   | Number of target variables (e.g., `ghi`, `dni`, `dhi`). |

---

#### Outputs

- `best_model.pth`: Model weights with lowest validation loss.  
- `training_curve.png`: Plot of training and validation losses.  
- `norm_stats.json`: Normalization statistics for reproducible preprocessing.  
- `checkpoint.pth`: Checkpoint file to resume training if interrupted.  

### 2. Validating the Model

After training, you can run the validation script to evaluate the model and save forecasts along with error metrics.

```bash
python validate.py
```

**What happens:**

- Loads the trained model (`best_model.pth`) and normalization stats (`norm_stats.json`).
- Runs inference on the validation split.
- Denormalizes predictions and targets.
- Computes forecast error metrics: MSE, MAE, RMSE.
- Saves all outputs in a compressed file `forecast_results.npz`.

## 3. Visualizing Forecast Results

The plotting script reads `forecast_results.npz` and generates figures for analysis.

```bash
python generate_plots_from_npz.py
```
### What happens

- **Error metrics** (MSE, MAE, RMSE) are plotted against forecast horizons.  
- **Actual vs predicted values** are plotted for selected horizons.  
- **All figures** are saved in the `plots/` directory.  

### Customizable options

- You can select which horizons to plot by modifying the `horizons_to_plot` variable inside the script.  
- The number of samples plotted can also be adjusted.


## 4. Dataset and Sample Visualization

The `dataset.py` script defines the `IrradianceForecastDataset` class, which prepares sequences of **sky images** and **historical time-series features** for training the forecasting model.

```bash
# Example usage for inspecting a single sample
python dataset.py
```
## What happens

### CSV loading & splitting
- Loads the dataset CSV file (`csv_path`) and splits it into **training** and **validation** sets.  
- The split ratio can be controlled with `val_ratio`.

### Normalization
- Computes mean and standard deviation of time-series features on the training set.  
- Normalizes input features using these statistics for consistency.  
- Validation split uses the saved normalization stats.

### Sequence construction
- Constructs sliding windows of past sky images (`img_seq_len`) and time-series steps (`ts_seq_len`) as inputs.  
- Targets are sequences of future irradiance values over a forecast horizon (`horizon`).

### Image preprocessing
- Resizes images to `img_size x img_size`, converts to tensors, and normalizes using standard ImageNet statistics.

### Returns per sample
- `sky_seq`: Tensor of sky images `[img_seq_len, C, H, W]`.  
- `ts_seq`: Tensor of normalized time-series inputs `[ts_seq_len, num_features]`.  
- `target_seq`: Tensor of normalized future targets `[horizon, target_dim]`.  
- `ts_time` / `tgt_time`: Corresponding timestamps for history and forecast sequences.

### Visualization (optional)
- Plots the sky image sequence.  
- Plots historical time-series inputs and forecast targets, un-normalized to physical units.  
- Marks the boundary between history and forecast for clarity.

<img src="images/sky_seq.png" alt="Sky Sequence" width="1000"/>
<img src="images/time_seq.png" alt="Time Sequence" width="1000"/>

---

## Customizable options

| Parameter | Description |
|-----------|-------------|
| `csv_path` | Path to the dataset CSV file. |
| `split` | `"train"` or `"val"` to select dataset split. |
| `img_seq_len` | Number of past sky images in each input sequence. |
| `ts_seq_len` | Number of past time-series steps in each input sequence. |
| `horizon` | Number of future steps to forecast. |
| `feature_cols` | List of input feature column names. Default: `["ghi", "dni", "dhi"]`. |
| `target_cols` | List of prediction target column names. Default: `["ghi"]`. |
| `img_size` | Size to which sky images are resized. Default: 224x224. |
| `normalization_stats` | Optional pre-computed mean/std for feature normalization. |

---

## Example Output

After running `python dataset.py`, a single sample is visualized:

- **Sky image sequence**: shows the past `img_seq_len` images.  
- **Time-series plot**: shows historical inputs and forecast targets.  
- **Timestamps**: labeled on x-axis, with history/forecast boundary marked.  

This allows quick inspection of how input sequences and forecast targets are structured before training the model.

