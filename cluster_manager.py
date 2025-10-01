#!/usr/bin/env python3

"""
cluster_manager.py - Centralized Dask cluster lifecycle management

This module provides utilities for creating, connecting to, and managing
Dask distributed clusters that persist across multiple processing operations.
"""

import os
import asyncio
from dask.distributed import Client, SSHCluster

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


def load_cluster_config(config_path: str = "config.toml") -> dict:
    """Load cluster configuration from TOML file"""
    config = {
        'use_dask': False,
        'ssh_hosts': ['localhost'],
        'ssh_workers_per_host': 1,
        'ssh_threads_per_worker': 16,
        'ssh_memory_per_worker': '16GB',
    }
    
    if config_path and os.path.exists(config_path):
        if tomllib is None:
            print(f"Warning: Cannot read {config_path} - tomli/tomllib not installed")
            return config
        
        try:
            with open(config_path, 'rb') as f:
                toml_config = tomllib.load(f)
                if 'cluster' in toml_config:
                    config.update(toml_config['cluster'])
                    print(f"Loaded cluster configuration from {config_path}")
        except Exception as e:
            print(f"Warning: Could not read config file: {e}")
    
    return config


async def create_dask_cluster(config: dict):
    """
    Create a new Dask SSH cluster based on configuration.
    
    Returns:
    --------
    client : dask.distributed.Client
        Dask client connected to cluster
    cluster : dask.distributed.SSHCluster
        Cluster object for lifecycle management
    """
    use_dask = config.get('use_dask', False)
    
    if not use_dask:
        print("Dask disabled - operations will use local processing")
        return None, None
    
    # Create SSH cluster
    ssh_hosts = config.get('ssh_hosts', ['localhost'])
    workers_per_host = config.get('ssh_workers_per_host', 1)
    threads_per_worker = config.get('ssh_threads_per_worker', 16)
    memory_per_worker = config.get('ssh_memory_per_worker', '16GB')
    
    print(f"\nCreating Dask SSH Cluster:")
    print(f"  Hosts: {ssh_hosts}")
    print(f"  Workers per host: {workers_per_host}")
    print(f"  Threads per worker: {threads_per_worker}")
    print(f"  Memory per worker: {memory_per_worker}")
    
    # Repeat hosts for multiple workers per host
    all_hosts = ssh_hosts * workers_per_host
    
    cluster = await SSHCluster(
        hosts=all_hosts,
        connect_options={"known_hosts": None},
        worker_options={
            "nthreads": threads_per_worker,
            "memory_limit": memory_per_worker,
        },
        scheduler_options={
            "port": 0,
            "dashboard_address": ":8797",
        },
        asynchronous=True
    )
    
    client = await Client(cluster, asynchronous=True)
    
    print(f"\n✓ Dask cluster ready!")
    print(f"  Dashboard: {client.dashboard_link}")
    print(f"  Scheduler: {client.scheduler.address}")
    print(f"  Workers: {len(client.scheduler_info()['workers'])}")
    print(f"  Total cores: {sum(w['nthreads'] for w in client.scheduler_info()['workers'].values())}")
    print()
    
    return client, cluster


async def connect_to_cluster(scheduler_address: str):
    """
    Connect to an existing Dask scheduler.
    
    Parameters:
    -----------
    scheduler_address : str
        Address of the Dask scheduler (e.g., 'tcp://10.0.1.2:8786')
    
    Returns:
    --------
    client : dask.distributed.Client
        Dask client connected to scheduler
    """
    print(f"Connecting to existing Dask scheduler: {scheduler_address}")
    client = await Client(scheduler_address, asynchronous=True)
    print(f"  ✓ Connected! Workers: {len(client.scheduler_info()['workers'])}")
    return client


async def shutdown_cluster(client, cluster):
    """
    Gracefully shutdown Dask cluster.
    
    Parameters:
    -----------
    client : dask.distributed.Client or None
        Client to close
    cluster : dask.distributed.Cluster or None
        Cluster to shutdown
    """
    if client:
        print("\nShutting down Dask client...")
        await client.close()
    
    if cluster:
        print("Shutting down Dask cluster...")
        await cluster.close()
        print("✓ Cluster shutdown complete")


def get_scheduler_address_from_client(client) -> str:
    """Extract scheduler address from client"""
    if client:
        return client.scheduler.address
    return ""
