# Local Development with Modal

This guide explains how to run the meshing pipeline on Modal directly from your local machine using your local code changes, bypassing the need to push to GitHub or deploy to AWS/Dev.

## Prerequisites

1.  **Modal CLI**: Ensure you have the `modal` package installed locally.
    ```bash
    pip install modal
    modal token new  # Authenticate with Modal
    ```
2.  **AWS Credentials**: Your local environment must have AWS credentials configured (typically in `~/.aws/credentials` or via env vars) to allow `boto3` to upload input files to S3.
    - Required permissions: `s3:PutObject` on `muaz-webdev-assets` (or your configured bucket).

## Running the Mesher Locally

The script `scripts/run_local_modal.py` mounts your local `backend/`, `core/`, `strategies/`, and `converters/` directories to the Modal container at runtime.

### Basic Usage

```bash
# Run with a local STEP file
modal run scripts/run_local_modal.py --input-file path/to/model.step
```

### Options

-   `--input-file`: Path to the local CAD file (`.step`, `.stp`). **Required**.
-   `--bucket`: S3 bucket to use for uploads/storage. Defaults to `S3_BUCKET_NAME` env var or `muaz-webdev-assets`.
-   `--strategy`: Meshing strategy to use (default: `tet_hxt`). Options: `tet_hxt`, `tet_delaunay`, `tet_frontal`, `hex_dominant`.
-   `--order`: Element order (default: `1`).

### Example

```bash
# Run a high-quality HXT mesh with element order 2
modal run scripts/run_local_modal.py --input-file cad_files/Cylinder.step --strategy tet_hxt --order 2
```

## How It Works

1.  **Upload**: The script uploads your local input file to `s3://<bucket>/uploads/local_dev/`.
2.  **Mount**: Modal mounts your current local project directories into the remote container.
3.  **Execute**: The `generate_mesh` function runs on Modal's GPUs using your *local* code.
4.  **stream**: Logs are streamed back to your terminal.
5.  **Result**: The final S3 paths for the mesh and result JSON are printed.

## Troubleshooting

-   **ModuleNotFoundError**: Ensure you run the command from the root of the repository (`MeshPackageLean/`).
-   **Auth Errors**: Run `modal token new` to refresh Modal credentials. Check AWS credentials for S3 upload failures.
