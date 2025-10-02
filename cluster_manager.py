#!/usr/bin/env python3

"""
cluster_manager.py - Centralized Dask cluster lifecycle management

This module provides utilities for creating, connecting to, and managing
Dask distributed clusters that persist across multiple processing operations.

"""

import os
import asyncio
import time
from pathlib import Path
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
    if isinstance(ssh_hosts, str):
        ssh_hosts = [ssh_hosts]

    if not ssh_hosts:
        raise ValueError("ssh_hosts must contain at least one entry")

    workers_per_host = max(int(config.get('ssh_workers_per_host', 1)), 1)
    threads_per_worker = config.get('ssh_threads_per_worker', 16)
    memory_per_worker = config.get('ssh_memory_per_worker', '16GB')

    scheduler_host = ssh_hosts[0]

    configured_worker_hosts = ssh_hosts[1:] if len(ssh_hosts) > 1 else []
    if not configured_worker_hosts:
        # No explicit worker hosts provided; schedule workers on the scheduler host
        configured_worker_hosts = [scheduler_host]

    worker_hosts: list[str] = []
    for host in configured_worker_hosts:
        worker_hosts.extend([host] * workers_per_host)

    hosts_for_cluster = [scheduler_host] + worker_hosts
    expected_workers = len(worker_hosts)

    print(f"\nCreating Dask SSH Cluster:")
    print(f"  Scheduler host: {scheduler_host}")
    print(f"  Worker hosts: {configured_worker_hosts}")
    print(f"  Workers per host: {workers_per_host}")
    print(f"  Threads per worker: {threads_per_worker}")
    print(f"  Memory per worker: {memory_per_worker}")

    # Get absolute path to worker preload script
    preload_script = str(Path(__file__).parent / "worker_preload.py")
    if os.path.exists(preload_script):
        print(f"  Using worker preload: {os.path.basename(preload_script)}")
        preload = [preload_script]
    else:
        print(f"  Warning: worker_preload.py not found, workers may have permission issues")
        preload = []

    preload_command = "import os; os.umask(0o000)"

    cluster = await SSHCluster(
        hosts=hosts_for_cluster,
        connect_options={"known_hosts": None},
        worker_options={
            "nthreads": threads_per_worker,
            "memory_limit": memory_per_worker,
            "preload": [preload_command],  # Run worker_preload.py on each worker startup
        },
        scheduler_options={
            "port": 0,
            "dashboard_address": ":8797",
            "preload": [preload_command],  # Also set on scheduler
        },
        asynchronous=True
    )

    client = await Client(cluster, asynchronous=True)

    # Wait for all workers to connect (with timeout)
    print(f"\n  Waiting for all {expected_workers} workers to connect...")
    max_wait = 60  # seconds
    start_time = time.time()

    while time.time() - start_time < max_wait:
        info = client.scheduler_info()
        num_workers = len(info['workers'])

        if num_workers >= expected_workers:
            break

        print(f"    {num_workers}/{expected_workers} workers connected...", flush=True)
        await asyncio.sleep(2)

    # Final check
    info = client.scheduler_info()
    num_workers = len(info['workers'])
    total_cores = sum(w['nthreads'] for w in info['workers'].values())

    print(f"\n✓ Dask cluster ready!")
    print(f"  Dashboard: {client.dashboard_link}")
    print(f"  Scheduler: {client.scheduler.address}")
    print(f"  Workers: {num_workers}/{expected_workers}")
    print(f"  Total cores: {total_cores}")

    if num_workers < expected_workers:
        print(f"  ⚠ Warning: Only {num_workers}/{expected_workers} workers connected")
        print(f"  Some remote nodes may be slow or unavailable")

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

    info = client.scheduler_info()
    print(f"  ✓ Connected! Workers: {len(info['workers'])}")
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
