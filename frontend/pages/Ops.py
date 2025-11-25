import streamlit as st
import requests
import os
import dotenv
import redis
import json
from datetime import datetime

dotenv.load_dotenv('.env')
API_URL = os.environ.get('API_URL', 'http://backend')
REDIS_URL = os.environ.get('REDIS_URL', 'redis://redis:6379')

st.set_page_config(page_title="Ops", layout="wide", page_icon="📊")

st.title("Redis Queue Monitor")
st.markdown("Real-time monitoring of the prediction queue and task status")

# Add auto-refresh
auto_refresh = st.checkbox("Auto-refresh (5 seconds)", value=False)
if auto_refresh:
    import time
    time.sleep(5)
    st.rerun()

# Connect to Redis
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=False)
    redis_client.ping()
    redis_connected = True
except Exception as e:
    redis_connected = False
    st.error(f"Cannot connect to Redis: {e}")
    st.stop()

# Create columns for metrics
col1, col2, col3, col4 = st.columns(4)

# Get queue statistics
try:
    queue_length = redis_client.llen("prediction_queue")
    
    # Get all result keys
    result_keys = redis_client.keys("result:*")
    total_results = len(result_keys)
    
    # Get Redis info
    redis_info = redis_client.info()
    used_memory = redis_info.get('used_memory_human', 'N/A')
    connected_clients = redis_info.get('connected_clients', 0)
    
    with col1:
        st.metric("Queue Length", queue_length, 
                 help="Number of tasks waiting to be processed")
    
    with col2:
        st.metric("Cached Results", total_results,
                 help="Number of completed results in cache (1hr TTL)")
    
    with col3:
        st.metric("Memory Used", used_memory,
                 help="Redis memory consumption")
    
    with col4:
        st.metric("Connected Clients", connected_clients,
                 help="Active Redis connections")
    
except Exception as e:
    st.error(f"Error fetching queue stats: {e}")

st.markdown("---")

# Two columns: Queue and Recent Results
col_queue, col_results = st.columns([1, 1])

with col_queue:
    st.subheader("📋 Pending Tasks Queue")
    
    if queue_length > 0:
        st.info(f"**{queue_length}** tasks in queue")
        
        # Show first few tasks in queue (without removing them)
        show_count = min(queue_length, 10)
        st.write(f"Showing first {show_count} tasks:")
        
        # Get tasks from queue without removing them
        tasks = redis_client.lrange("prediction_queue", 0, show_count - 1)
        
        for idx, task_bytes in enumerate(tasks, 1):
            try:
                task_data = json.loads(task_bytes)
                task_id = task_data.get('task_id', 'Unknown')
                task_type = task_data.get('task_type', 'animal')
                
                with st.expander(f"#{idx} - {task_type.upper()} - {task_id[:8]}..."):
                    st.json(task_data)
            except Exception as e:
                st.error(f"Error parsing task {idx}: {e}")
        
        if queue_length > show_count:
            st.info(f"... and {queue_length - show_count} more tasks")
    else:
        st.success("Queue is empty - no pending tasks")

with col_results:
    st.subheader("Recent Results")
    
    if total_results > 0:
        st.info(f"**{total_results}** results cached")
        
        # Show recent results
        show_count = min(total_results, 10)
        st.write(f"Showing latest {show_count} results:")
        
        # Sort keys by timestamp (newest first)
        result_keys_sorted = sorted(result_keys, reverse=True)[:show_count]
        
        for key in result_keys_sorted:
            try:
                task_id = key.decode('utf-8').replace('result:', '')
                result_data = redis_client.get(key)
                result_json = json.loads(result_data)
                
                # Get TTL (time to live)
                ttl = redis_client.ttl(key)
                ttl_minutes = ttl // 60 if ttl > 0 else 0
                
                # Determine task type
                if 'sketch' in result_json:
                    task_type = "🎨 Sketch"
                    success = result_json.get('sketch', {}).get('success', False)
                else:
                    task_type = "🐾 Animal"
                    # Check if any version succeeded
                    success = any(v.get('success', False) for v in result_json.values())
                
                status_icon = "✅" if success else "❌"
                
                with st.expander(f"{status_icon} {task_type} - {task_id[:8]}... (expires in {ttl_minutes}m)"):
                    st.json(result_json)
            except Exception as e:
                st.error(f"Error parsing result: {e}")
        
        if total_results > show_count:
            st.info(f"... and {total_results - show_count} more results")
    else:
        st.info("No cached results yet")

st.markdown("---")

# Task Search
st.subheader("Search Task by ID")
search_col1, search_col2 = st.columns([3, 1])

with search_col1:
    task_id_search = st.text_input("Enter Task ID", placeholder="e.g., 123e4567-e89b-12d3-a456-426614174000")

with search_col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    search_button = st.button("Search", type="primary", use_container_width=True)

if search_button and task_id_search:
    try:
        # Check if task is in queue
        tasks = redis_client.lrange("prediction_queue", 0, -1)
        in_queue = False
        queue_position = None
        
        for idx, task_bytes in enumerate(tasks, 1):
            task_data = json.loads(task_bytes)
            if task_data.get('task_id') == task_id_search:
                in_queue = True
                queue_position = idx
                break
        
        if in_queue:
            st.warning(f"Task is in queue at position #{queue_position}")
            st.info(f"Estimated wait: ~{queue_position * 2} seconds (assuming 2s per task)")
        else:
            # Check if result exists
            result_key = f"result:{task_id_search}"
            result_data = redis_client.get(result_key)
            
            if result_data:
                result_json = json.loads(result_data)
                ttl = redis_client.ttl(result_key)
                ttl_minutes = ttl // 60 if ttl > 0 else 0
                
                st.success(f"Task completed! Result expires in {ttl_minutes} minutes")
                
                # Show results
                if 'sketch' in result_json:
                    st.subheader("🎨 Sketch Recognition Result")
                    sketch_result = result_json['sketch']
                    if sketch_result.get('success'):
                        predictions = sketch_result.get('predictions', [])
                        if predictions:
                            st.metric("Top Prediction", 
                                    predictions[0]['category'].upper(),
                                    f"{predictions[0]['confidence']:.1f}%")
                            st.json(predictions[:5])
                    else:
                        st.error(f"Error: {sketch_result.get('error')}")
                else:
                    st.subheader("🐾 Animal Classification Result")
                    for version, result in result_json.items():
                        with st.expander(f"Version {version.upper()}"):
                            if result.get('success'):
                                st.json(result)
                            else:
                                st.error(f"Error: {result.get('error')}")
            else:
                st.error("Task ID not found. It may have expired or never existed.")
                st.info("Results are cached for 1 hour after completion.")
    
    except Exception as e:
        st.error(f"Error searching for task: {e}")

st.markdown("---")

# Refresh button
st.markdown("---")
if st.button("🔄 Refresh Now", use_container_width=True):
    st.rerun()
