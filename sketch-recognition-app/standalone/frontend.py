import streamlit as st
import streamlit.components.v1 as components
import requests
import os

# Page configuration
st.set_page_config(
    page_title="Sketch Recognition App",
    page_icon="><",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FastAPI endpoint - Assumes running locally on port 8000
# In a real production setup, this should be the public URL of the API
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #4ECDC4;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<div class="main-header"> Sketch Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Draw a sketch and get instant AI predictions</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ About")
        st.write("""
        AI-powered sketch recognition using a CNN model trained on the QuickDraw dataset.
        
        **How to use:**
        1. Draw a sketch in the canvas
        2. Click Predict
        3. See results instantly
        4. Click Clear to start over
        """)
        
        st.divider()
        
        st.header(" Supported Categories")
        categories = [
            'apple', 'banana', 'cat', 'dog', 'house',
            'tree', 'car', 'fish', 'bird', 'clock',
            'book', 'chair', 'cup', 'star', 'heart',
            'smiley face', 'sun', 'moon', 'key', 'hammer'
        ]
        
        cols = st.columns(2)
        for i, cat in enumerate(categories):
            with cols[i % 2]:
                st.write(f"• {cat.title()}")
        
        st.divider()
        
        st.header(" Model Info")
        st.write("**Framework:** TensorFlow/Keras")
        st.write("**Classes:** 20 categories")
        st.write("**Platform:** Docker Standalone")
    
    # Main content - Full width canvas and predictions below
    st.header(" Draw Your Sketch")
    
    # HTML Canvas with direct API call
    canvas_html = f"""
    <div style="display: flex; flex-direction: row; justify-content: center; gap: 40px; flex-wrap: wrap; align-items: flex-start;">
        <div style="text-align: center;">
            <canvas id="canvas" width="400" height="400" style="border: 2px solid #333; cursor: crosshair; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);"></canvas>
            <br><br>
            <div style="display: flex; justify-content: center; gap: 15px;">
                <button onclick="clearCanvas()" style="padding: 12px 24px; font-size: 16px; cursor: pointer; background: #ff4444; color: white; border: none; border-radius: 5px; font-weight: bold;">Clear Canvas</button>
                <button onclick="predictSketch()" style="padding: 12px 24px; font-size: 16px; cursor: pointer; background: #4CAF50; color: white; border: none; border-radius: 5px; font-weight: bold;"> Predict</button>
            </div>
            <div id="status" style="margin-top: 15px; font-size: 16px; color: #555; font-weight: 500; min-height: 24px;"></div>
        </div>
        
        <div id="results" style="text-align: left; width: 350px; min-height: 400px;">
            <div style="background: #f8f9fa; border-radius: 10px; padding: 25px; border: 1px solid #dee2e6; height: 100%; box-sizing: border-box;">
                <h3 style="color: #6c757d; margin-top: 0; text-align: center;">Predictions</h3>
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 300px; color: #adb5bd;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                    <p style="margin-top: 10px;">Draw something and click Predict</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let drawing = false;
        
        // Set up canvas
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 20;
        ctx.lineCap = 'round';
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Mouse events
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);
        
        // Touch events
        canvas.addEventListener('touchstart', handleTouchStart);
        canvas.addEventListener('touchmove', handleTouchMove);
        canvas.addEventListener('touchend', stopDrawing);
        
        function startDrawing(e) {{
            drawing = true;
            const rect = canvas.getBoundingClientRect();
            ctx.beginPath();
            ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
        }}
        
        function draw(e) {{
            if (!drawing) return;
            const rect = canvas.getBoundingClientRect();
            ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
            ctx.stroke();
        }}
        
        function stopDrawing() {{
            drawing = false;
        }}
        
        function handleTouchStart(e) {{
            e.preventDefault();
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            drawing = true;
            ctx.beginPath();
            ctx.moveTo(touch.clientX - rect.left, touch.clientY - rect.top);
        }}
        
        function handleTouchMove(e) {{
            e.preventDefault();
            if (!drawing) return;
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            ctx.lineTo(touch.clientX - rect.left, touch.clientY - rect.top);
            ctx.stroke();
        }}
        
        function clearCanvas() {{
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('results').innerHTML = `
                <div style="background: #f8f9fa; border-radius: 10px; padding: 25px; border: 1px solid #dee2e6; height: 100%; box-sizing: border-box;">
                    <h3 style="color: #6c757d; margin-top: 0; text-align: center;">Predictions</h3>
                    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 300px; color: #adb5bd;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                        <p style="margin-top: 10px;">Draw something and click Predict</p>
                    </div>
                </div>`;
            document.getElementById('status').innerHTML = '';
        }}
        
        async function predictSketch() {{
            const statusDiv = document.getElementById('status');
            const resultsDiv = document.getElementById('results');
            
            statusDiv.innerHTML = ' Analyzing your sketch...';
            
            try {{
                // Get canvas data as base64
                const imageData = canvas.toDataURL('image/png').split(',')[1];
                
                // Call FastAPI endpoint
                const response = await fetch('{API_URL}/predict', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{ image: imageData }})
                }});
                
                if (response.ok) {{
                    const data = await response.json();
                    statusDiv.innerHTML = ' Prediction Complete!';
                    
                    // Display results
                    let html = '<div style="background: #ffffff; padding: 25px; border-radius: 10px; border: 1px solid #e9ecef; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">';
                    html += '<h3 style="color: #FF6B6B; margin-top: 0; border-bottom: 2px solid #f0f0f0; padding-bottom: 10px;"> Top Prediction</h3>';
                    html += `<div style="text-align: center; margin: 20px 0;">`;
                    html += `<h1 style="margin: 0; font-size: 36px; color: #333;">${{data.predictions[0].category.toUpperCase()}}</h1>`;
                    html += `<p style="font-size: 20px; color: #4CAF50; font-weight: bold; margin: 5px 0;">${{data.predictions[0].confidence.toFixed(1)}}%</p>`;
                    html += `</div>`;
                    
                    html += '<h4 style="color: #666; margin-top: 25px; margin-bottom: 15px;">Other Possibilities:</h4>';
                    
                    data.predictions.slice(1, 5).forEach((pred, i) => {{
                        html += `<div style="margin: 12px 0;">`;
                        html += `<div style="display: flex; justify-content: space-between; margin-bottom: 4px;">`;
                        html += `<span style="font-weight: 500; color: #444;">${{pred.category.charAt(0).toUpperCase() + pred.category.slice(1)}}</span>`;
                        html += `<span style="color: #666;">${{pred.confidence.toFixed(1)}}%</span>`;
                        html += `</div>`;
                        html += `<div style="background: #eee; height: 8px; border-radius: 4px; overflow: hidden;"><div style="background: #4CAF50; height: 100%; width: ${{pred.confidence}}%;"></div></div>`;
                        html += `</div>`;
                    }});
                    
                    html += '</div>';
                    resultsDiv.innerHTML = html;
                }} else {{
                    statusDiv.innerHTML = ' Error: ' + response.status;
                    resultsDiv.innerHTML = '<p style="color: red;">Failed to get prediction. Please try again.</p>';
                }}
            }} catch (error) {{
                statusDiv.innerHTML = ' Error occurred';
                resultsDiv.innerHTML = '<p style="color: red;">Error: ' + error.message + '</p>';
            }}
        }}
    </script>
    """
    
    # Display the canvas
    components.html(canvas_html, height=700, scrolling=False)
    
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>Built with ❤️ using Streamlit, TensorFlow, and Docker</p>
        <p>Dataset: Google QuickDraw</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
