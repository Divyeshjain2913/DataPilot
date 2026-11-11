import sys
sys.path.insert(0, ".")
import streamlit as st
from utils.helpers import render_html

html_content = """
<div class="ent-box">
    <div class="ent-box-header">
        <h4 class="ent-box-title">Quality Scorecard: 99.5%</h4>
        <span class="status-pill live">Grade A+</span>
    </div>
    <div class="goal-item">
        <div class="goal-header">
            <span class="goal-label">Completeness Health</span>
            <span class="goal-val">99.6%</span>
        </div>
        <div class="goal-track"><div class="goal-bar" style="width: 99.6%; background-color: #38BDF8;"></div></div>
    </div>
</div>
"""

print("Testing st.html directly...")
st.html(html_content)
print("st.html executed without error!")
