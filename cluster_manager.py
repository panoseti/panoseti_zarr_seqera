#!/usr/bin/env python3
"""
Shared Dask cluster management for PANOSETI pipeline.

Supports multiple cluster types:
- SSHCluster: Manually configured SSH hosts
- SLURMCluster: SLURM-managed cluster
- LocalCluster: Single-node local cluster
- External: Connect to existing scheduler

The cluster persists across pipeline steps unless explicitly closed.
"""

import os
from typing import Optional, Tuple
from dask.distributed import Client, SSHCluster, LocalCluster
from dask_jobqueue import SLURMCluster

_GLOBAL_CLIENT = None
_GLOBAL_CLUSTER = None

class ClusterType:
    SSH = "ssh"
    SLURM = "slurm"
    LOCAL = "local"
    EXTERNAL = "external"
    DISABLED = "disabled"

async def get_or_create_cluster(config: dict, force_new: bool = False) -> Tuple[Optional[Client], Optional[object]]:
    """
    Get existing cluster or create new one based on configuration.
    
    Parameters:
    -----------
    config : dict
        Configuration dict with cluster settings
    force_new : bool
        If True, create new cluster even if one exists
        
    Returns:
    --------
    client : Client or None
        Dask client connected to cluster
    cluster : Cluster or None
        Cluster object if created, None if external or disabled
    """
    global _GLOBAL_CLIENT, _GLOBAL_CLUSTER
    
    # Return existing cluster if available and not forcing new
    if not force_new and _GLOBAL_CLIENT is not None:
        try:
            # Check if client is still valid
            _GLOBAL_CLIENT.scheduler_info()
            print("\nReusing existing Dask cluster")
            print(f"  Workers: {len(_GLOBAL_CLIENT.scheduler_info()['workers'])}")
            print(f"  Dashboard: {_GLOBAL_CLIENT.dashboard_link}\n")
            return _GLOBAL_CLIENT, _GLOBAL_CLUSTER
        except Exception as e:
            print(f"Warning: Existing cluster is no longer valid: {e}")
            _GLOBAL_CLIENT = None
            _GLOBAL_CLUSTER = None
    
    use_dask = config.get('use_dask', False)
    if not use_dask:
        print("Dask disabled - using local threading\n")
        return None, None
    
    # Determine cluster type
    cluster_type = config.get('cluster_type', 'ssh').lower()
    scheduler_address = config.get('dask_scheduler_address', '')
    
    # External scheduler takes precedence
    if scheduler_address:
        cluster_type = ClusterType.EXTERNAL
    
    print(f"\nInitializing Dask cluster (type: {cluster_type})...")
    
    if cluster_type == ClusterType.EXTERNAL:
        client, cluster = await _connect_external(scheduler_address)
    elif cluster_type == ClusterType.SSH:
        client, cluster = await _create_ssh_cluster(config)
    elif cluster_type == ClusterType.SLURM:
        client, cluster = await _create_slurm_cluster(config)
    elif cluster_type == ClusterType.LOCAL:
        client, cluster = await _create_local_cluster(config)
    else:
        raise ValueError(f"Unknown cluster type: {cluster_type}")
    
    # Cache globally
    _GLOBAL_CLIENT = client
    _GLOBAL_CLUSTER = cluster
    
    return client, cluster

async def _connect_external(scheduler_address: str) -> Tuple[Client, None]:
    """Connect to existing Dask scheduler"""
    print(f"  Connecting to existing scheduler: {scheduler_address}")
    client = await Client(scheduler_address, asynchronous=True)
    print(f"  ✓ Connected! Workers: {len(client.scheduler_info()['workers'])}")
    print(f"  Dashboard: {client.dashboard_link}\n")
    return client, None

async def _create_ssh_cluster(config: dict) -> Tuple[Client, SSHCluster]:
    """Create SSH-based cluster"""
    ssh_hosts = config.get('ssh_hosts', ['localhost'])
    workers_per_host = config.get('ssh_workers_per_host', 1)
    threads_per_worker = config.get('ssh_threads_per_worker', 8)
    memory_per_worker = config.get('ssh_memory_per_worker', '16GB')
    
    print(f"  Creating SSH Cluster:")
    print(f"    Hosts: {ssh_hosts}")
    print(f"    Workers per host: {workers_per_host}")
    print(f"    Threads per worker: {threads_per_worker}")
    print(f"    Memory per worker: {memory_per_worker}")
    
    cluster = await SSHCluster(
        hosts=ssh_hosts * workers_per_host,
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
    
    print(f"  ✓ SSH Cluster ready!")
    print(f"    Dashboard: {client.dashboard_link}")
    print(f"    Workers: {len(client.scheduler_info()['workers'])}")
    print(f"    Total cores: {sum(w['nthreads'] for w in client.scheduler_info()['workers'].values())}\n")
    
    return client, cluster

async def _create_slurm_cluster(config: dict) -> Tuple[Client, SLURMCluster]:
    """Create SLURM-based cluster"""
    slurm_cores = config.get('slurm_cores', 8)
    slurm_processes = config.get('slurm_processes', 1)
    slurm_memory = config.get('slurm_memory', '16GB')
    slurm_walltime = config.get('slurm_walltime', '01:00:00')
    slurm_queue = config.get('slurm_queue', None)
    slurm_account = config.get('slurm_account', None)
    slurm_workers = config.get('slurm_workers', 4)
    
    print(f"  Creating SLURM Cluster:")
    print(f"    Cores per job: {slurm_cores}")
    print(f"    Processes per job: {slurm_processes}")
    print(f"    Memory per job: {slurm_memory}")
    print(f"    Walltime: {slurm_walltime}")
    if slurm_queue:
        print(f"    Queue: {slurm_queue}")
    if slurm_account:
        print(f"    Account: {slurm_account}")
    
    cluster_kwargs = {
        'cores': slurm_cores,
        'processes': slurm_processes,
        'memory': slurm_memory,
        'walltime': slurm_walltime,
        'asynchronous': True,
    }
    
    if slurm_queue:
        cluster_kwargs['queue'] = slurm_queue
    if slurm_account:
        cluster_kwargs['account'] = slurm_account
    
    cluster = SLURMCluster(**cluster_kwargs)
    cluster.scale(slurm_workers)
    
    client = await Client(cluster, asynchronous=True)
    
    print(f"  ✓ SLURM Cluster created!")
    print(f"    Scaling to {slurm_workers} workers...")
    print(f"    Dashboard: {client.dashboard_link}\n")
    
    return client, cluster

async def _create_local_cluster(config: dict) -> Tuple[Client, LocalCluster]:
    """Create local single-node cluster"""
    n_workers = config.get('local_n_workers', None)
    threads_per_worker = config.get('local_threads_per_worker', 1)
    memory_limit = config.get('local_memory_limit', 'auto')
    
    print(f"  Creating Local Cluster:")
    print(f"    Workers: {n_workers or 'auto'}")
    print(f"    Threads per worker: {threads_per_worker}")
    print(f"    Memory limit: {memory_limit}")
    
    cluster = await LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        dashboard_address=':8787',
        asynchronous=True
    )
    
    client = await Client(cluster, asynchronous=True)
    
    print(f"  ✓ Local Cluster ready!")
    print(f"    Dashboard: {client.dashboard_link}")
    print(f"    Workers: {len(client.scheduler_info()['workers'])}\n")
    
    return client, cluster

async def close_cluster():
    """Close the global cluster and client"""
    global _GLOBAL_CLIENT, _GLOBAL_CLUSTER
    
    if _GLOBAL_CLIENT:
        try:
            await _GLOBAL_CLIENT.close()
            print("  ✓ Closed Dask client")
        except Exception as e:
            print(f"  Warning: Error closing client: {e}")
        _GLOBAL_CLIENT = None
    
    if _GLOBAL_CLUSTER:
        try:
            await _GLOBAL_CLUSTER.close()
            print("  ✓ Closed Dask cluster")
        except Exception as e:
            print(f"  Warning: Error closing cluster: {e}")
        _GLOBAL_CLUSTER = None

def get_current_client() -> Optional[Client]:
    """Get the current global client if it exists"""
    return _GLOBAL_CLIENT
