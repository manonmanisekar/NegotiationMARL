import os, subprocess, glob
scripts = ['kpi_analysis.py']
base = os.path.dirname(__file__)
res = os.path.join(base,'..','results') if os.path.exists(os.path.join(base,'..','results')) else 'results'
os.makedirs(os.path.join(base,'..','results','analysis'), exist_ok=True)
for s in scripts:
    p = os.path.join(base,s)
    subprocess.run(['python', p], check=False)
print('Analysis complete. Check results/analysis for outputs.')
