# Happy Path Validation Checklist

This checklist documents the 20+ steps that are automatically validated by `scripts/validate_happy_path.py`.

## System Health
- [ ] **System Health Check**: Verify `/api/health` returns 200 and "healthy".
- [ ] **Strategy Listing**: Verify `/api/strategies` returns essential strategies (e.g., `tetrahedral_hxt`).

## Authentication & User
- [ ] **User Registration**: Create a new test user.
- [ ] **User Login**: Authenticate and retrieve a JWT access token.
- [ ] **Token Validation**: Verify the token grants access to protected routes.
- [ ] **User Profile**: Verify user profile data (e.g., storage quota) works.

## Project & File Management
- [ ] **File Upload**: Upload a standard `.step` file.
- [ ] **Upload Response**: Verify successful upload response with Project ID.
- [ ] **Project Metadata**: Verify project details (file size, hash) are correct.
- [ ] **Preview Generation**: Verify a 3D preview is generated for the file.
- [ ] **Preview Availability**: Verify preview data/file is accessible.

## Mesh Generation (Local)
- [ ] **Job Submission (Local)**: Submit a mesh job using the local CPU provider.
- [ ] **Job ID Creation**: Verify a unique Job ID is returned.
- [ ] **Job Monitoring**: Monitor the job status from `processing` to `completed`.
- [ ] **Job Logs**: Verify that real-time logs are available.
- [ ] **Job Completion**: Confirm the job finishes successfully.
- [ ] **Result Metadata**: Verify node/element counts are recorded.
- [ ] **Output Path**: Verify the output mesh file path is generated.

## Mesh Generation (Cloud/Modal)
- [ ] **Modal Configuration**: Check if Modal cloud compute is enabled/configured.
- [ ] **Job Submission (Modal)**: Submit a mesh job explicitly using Modal.
- [ ] **Modal Job Monitoring**: Monitor the cloud job status.
- [ ] **Modal Result Verification**: Verify cloud job results and output.

## Cleanup
- [ ] **Project Cleanup**: Delete the test project.
- [ ] **User Cleanup**: Delete or cleanup the test user.
