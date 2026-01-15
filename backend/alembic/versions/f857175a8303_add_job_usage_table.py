"""add_job_usage_table

Revision ID: f857175a8303
Revises: f1836ef5e842
Create Date: 2026-01-13 01:13:35.080361

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f857175a8303'
down_revision: Union[str, Sequence[str], None] = 'f1836ef5e842'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Check if job_usage table already exists (created by db.create_all())
    from sqlalchemy import inspect
    conn = op.get_bind()
    inspector = inspect(conn)
    tables = inspector.get_table_names()
    
    if 'job_usage' not in tables:
        # Create job_usage table
        op.create_table(
            'job_usage',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('job_id', sa.String(length=100), nullable=True),
            sa.Column('job_type', sa.String(length=50), nullable=False),
            sa.Column('started_at', sa.DateTime(), nullable=False),
            sa.Column('completed_at', sa.DateTime(), nullable=True),
            sa.Column('status', sa.String(length=20), nullable=False),
            sa.Column('compute_backend', sa.String(length=20), nullable=True),
            sa.Column('project_id', sa.String(length=36), nullable=True),
            sa.Column('batch_id', sa.String(length=36), nullable=True),
            sa.Column('batch_job_id', sa.String(length=36), nullable=True),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
            sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
            sa.ForeignKeyConstraint(['batch_id'], ['batches.id'], ),
            sa.ForeignKeyConstraint(['batch_job_id'], ['batch_jobs.id'], ),
            sa.PrimaryKeyConstraint('id')
        )
    
    # Create indexes if they don't exist (only if table exists)
    if 'job_usage' in tables:
        indexes = inspector.get_indexes('job_usage')
        index_names = [idx['name'] for idx in indexes]
        
        if 'ix_job_usage_user_id' not in index_names:
            op.create_index(op.f('ix_job_usage_user_id'), 'job_usage', ['user_id'], unique=False)
        if 'ix_job_usage_started_at' not in index_names:
            op.create_index(op.f('ix_job_usage_started_at'), 'job_usage', ['started_at'], unique=False)
        if 'ix_job_usage_status' not in index_names:
            op.create_index(op.f('ix_job_usage_status'), 'job_usage', ['status'], unique=False)
        if 'ix_job_usage_job_id' not in index_names:
            op.create_index(op.f('ix_job_usage_job_id'), 'job_usage', ['job_id'], unique=False)
        if 'ix_job_usage_user_started' not in index_names:
            op.create_index('ix_job_usage_user_started', 'job_usage', ['user_id', 'started_at'], unique=False)
    
    # Add last_api_request_at to users table for time online tracking
    users_columns = [col['name'] for col in inspector.get_columns('users')]
    if 'last_api_request_at' not in users_columns:
        op.add_column('users', sa.Column('last_api_request_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('users', 'last_api_request_at')
    op.drop_index('ix_job_usage_user_started', table_name='job_usage')
    op.drop_index(op.f('ix_job_usage_job_id'), table_name='job_usage')
    op.drop_index(op.f('ix_job_usage_status'), table_name='job_usage')
    op.drop_index(op.f('ix_job_usage_started_at'), table_name='job_usage')
    op.drop_index(op.f('ix_job_usage_user_id'), table_name='job_usage')
    op.drop_table('job_usage')
