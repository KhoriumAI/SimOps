"""
Storage utilities - supports both local filesystem and AWS S3
Folder structure for S3: {user_email}/uploads/ and {user_email}/mesh/
"""
import os
from pathlib import Path
from flask import current_app

# boto3 is optional - only required for S3 storage (production)
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    ClientError = Exception  # Fallback for type hints


class StorageBackend:
    """Abstract storage backend"""
    
    def save_file(self, file_data, filename: str, user_email: str = None, file_type: str = 'uploads') -> str:
        raise NotImplementedError
    
    def get_file(self, filepath: str) -> bytes:
        raise NotImplementedError
    
    def delete_file(self, filepath: str) -> bool:
        raise NotImplementedError
    
    def file_exists(self, filepath: str) -> bool:
        raise NotImplementedError
    
    def get_file_url(self, filepath: str, expires_in: int = 3600) -> str:
        raise NotImplementedError
    
    def download_to_local(self, remote_path: str, local_path: str) -> str:
        """Download file to local filesystem (for processing)"""
        raise NotImplementedError
    
    def save_local_file(self, local_path: str, filename: str, user_email: str = None, file_type: str = 'mesh') -> str:
        """Upload a local file to storage"""
        raise NotImplementedError


class LocalStorage(StorageBackend):
    """
    Local filesystem storage for development
    
    Keeps the same simple flat structure as before:
    - uploads/{project_id}_{filename}
    - outputs/{mesh_file}
    
    user_email and file_type parameters are ignored to maintain backward compatibility
    """
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_file(self, file_data, filename: str, user_email: str = None, file_type: str = 'uploads') -> str:
        """
        Save file to local filesystem
        Note: user_email is ignored in local storage to keep existing behavior
        """
        # Use base_path directly (no email subfolders in local dev)
        self.base_path.mkdir(parents=True, exist_ok=True)
        filepath = self.base_path / filename
        
        if hasattr(file_data, 'save'):
            # Flask FileStorage object
            file_data.save(str(filepath))
        elif hasattr(file_data, 'read'):
            # File-like object
            with open(filepath, 'wb') as f:
                f.write(file_data.read())
        else:
            # Raw bytes
            with open(filepath, 'wb') as f:
                f.write(file_data)
        
        return str(filepath)
    
    def get_file(self, filepath: str) -> bytes:
        with open(filepath, 'rb') as f:
            return f.read()
    
    def delete_file(self, filepath: str) -> bool:
        try:
            Path(filepath).unlink()
            return True
        except Exception:
            return False
    
    def file_exists(self, filepath: str) -> bool:
        return Path(filepath).exists()
    
    def get_file_url(self, filepath: str, expires_in: int = 3600) -> str:
        """
        For local storage, returns the file path.
        
        NOTE: Browsers cannot access file:// URLs for security reasons.
        In development, use a Flask route to serve files instead:
            @app.route('/files/<path:filename>')
            def serve_file(filename):
                return send_from_directory(UPLOAD_FOLDER, filename)
        """
        return filepath
    
    def download_to_local(self, remote_path: str, local_path: str) -> str:
        """
        Pass-through for local storage - file is already local.
        Just copies to the target location if different, or returns as-is.
        """
        import shutil
        
        remote_path = str(remote_path)
        local_path = str(local_path)
        
        # If same path, nothing to do
        if Path(remote_path).resolve() == Path(local_path).resolve():
            return local_path
        
        # Copy to target location
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote_path, local_path)
        return local_path
    
    def save_local_file(self, local_path: str, filename: str, user_email: str = None, file_type: str = 'mesh') -> str:
        """
        For local storage, just copy/move the file to the storage directory.
        """
        import shutil
        
        dest_path = self.base_path / filename
        
        # If same path, nothing to do
        if Path(local_path).resolve() == dest_path.resolve():
            return str(dest_path)
        
        shutil.copy2(local_path, dest_path)
        return str(dest_path)


class S3Storage(StorageBackend):
    """
    AWS S3 storage with user email-based folder structure
    Structure: s3://muaz-webdev-assets/{user_email}/uploads/{filename}
               s3://muaz-webdev-assets/{user_email}/mesh/{filename}
    
    Requires boto3 to be installed: pip install boto3
    """
    
    def __init__(self, bucket_name: str, region: str = 'us-west-1', 
                 access_key: str = None, secret_key: str = None):
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for S3 storage. Install it with: pip install boto3"
            )
        self.bucket_name = bucket_name
        self.region = region
        
        # Create S3 client
        if access_key and secret_key:
            self.s3_client = boto3.client(
                's3',
                region_name=region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )
        else:
            # Use IAM role or environment credentials
            self.s3_client = boto3.client('s3', region_name=region)
    
    def _get_s3_key(self, filename: str, user_email: str = None, file_type: str = 'uploads') -> str:
        """
        Build S3 key: {user_email}/{file_type}/{filename}
        Example: john@example.com/uploads/model.step
                 john@example.com/mesh/model_mesh.msh
        """
        if user_email:
            return f"{user_email}/{file_type}/{filename}"
        return f"{file_type}/{filename}"
    
    def save_file(self, file_data, filename: str, user_email: str = None, file_type: str = 'uploads') -> str:
        """
        Save file to S3 with user email folder structure
        
        Args:
            file_data: File content (FileStorage, file-like object, or bytes)
            filename: Name of the file
            user_email: User's email for folder organization
            file_type: 'uploads' for CAD files, 'mesh' for generated meshes
        
        Returns:
            S3 URI: s3://bucket/user@email.com/uploads/filename
        """
        s3_key = self._get_s3_key(filename, user_email, file_type)
        
        try:
            if hasattr(file_data, 'read'):
                # File-like object - reset position first
                if hasattr(file_data, 'seek'):
                    file_data.seek(0)
                self.s3_client.upload_fileobj(file_data, self.bucket_name, s3_key)
            else:
                # Raw bytes
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=file_data
                )
            
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            print(f"[S3] Uploaded: {s3_uri}")
            return s3_uri
        except ClientError as e:
            print(f"[S3 ERROR] Failed to upload {filename}: {e}")
            raise
    
    def save_local_file(self, local_path: str, filename: str, user_email: str = None, file_type: str = 'mesh') -> str:
        """Upload a local file to S3"""
        s3_key = self._get_s3_key(filename, user_email, file_type)
        
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            print(f"[S3] Uploaded local file: {s3_uri}")
            return s3_uri
        except ClientError as e:
            print(f"[S3 ERROR] Failed to upload {local_path}: {e}")
            raise
    
    def get_file(self, filepath: str) -> bytes:
        """Download file content from S3"""
        bucket, key = self._parse_s3_uri(filepath)
        
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
        except ClientError as e:
            print(f"[S3 ERROR] Failed to get {filepath}: {e}")
            raise
    
    def delete_file(self, filepath: str) -> bool:
        """Delete file from S3"""
        bucket, key = self._parse_s3_uri(filepath)
        
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            print(f"[S3] Deleted: {filepath}")
            return True
        except ClientError as e:
            print(f"[S3 ERROR] Failed to delete {filepath}: {e}")
            return False
    
    def file_exists(self, filepath: str) -> bool:
        """Check if file exists in S3"""
        bucket, key = self._parse_s3_uri(filepath)
        
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False
    
    def get_file_url(self, filepath: str, expires_in: int = 3600) -> str:
        """Generate a presigned URL for downloading (valid for expires_in seconds)"""
        bucket, key = self._parse_s3_uri(filepath)
        
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expires_in
            )
            return url
        except ClientError as e:
            print(f"[S3 ERROR] Failed to generate URL for {filepath}: {e}")
            raise
    
    def download_to_local(self, s3_path: str, local_path: str) -> str:
        """Download S3 file to local filesystem (for mesh processing)"""
        bucket, key = self._parse_s3_uri(s3_path)
        
        print(f"[S3 DEBUG] Downloading from Bucket='{bucket}', Key='{key}'")
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self.s3_client.download_file(bucket, key, local_path)
        print(f"[S3] Downloaded {s3_path} to {local_path}")
        return local_path
    
    def _parse_s3_uri(self, filepath: str) -> tuple:
        """Parse S3 URI or key into (bucket, key)"""
        if filepath.startswith('s3://'):
            # s3://bucket/key format
            parts = filepath.replace('s3://', '').split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ''
        else:
            # Assume it's just the key
            bucket = self.bucket_name
            key = filepath
        return bucket, key
    
    def list_user_files(self, user_email: str, file_type: str = None) -> list:
        """List all files for a user"""
        prefix = f"{user_email}/"
        if file_type:
            prefix = f"{user_email}/{file_type}/"
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            files = []
            for obj in response.get('Contents', []):
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat(),
                    'uri': f"s3://{self.bucket_name}/{obj['Key']}"
                })
            return files
        except ClientError as e:
            print(f"[S3 ERROR] Failed to list files for {user_email}: {e}")
            return []


def get_storage() -> StorageBackend:
    """Factory function to get the appropriate storage backend"""
    if current_app.config.get('USE_S3'):
        return S3Storage(
            bucket_name=current_app.config['S3_BUCKET_NAME'],
            region=current_app.config.get('AWS_REGION', 'us-west-1'),
            access_key=current_app.config.get('AWS_ACCESS_KEY_ID'),
            secret_key=current_app.config.get('AWS_SECRET_ACCESS_KEY')
        )
    else:
        return LocalStorage(current_app.config['UPLOAD_FOLDER'])
