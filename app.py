"""Streamlit interface for the educational reduced-order Model 1 PWR core."""
import streamlit as st
import plotly.graph_objects as go
from models.reactor_model_1 import ReactorParameters
from models.controller import ControlParameters
from models.disturbances import Scenario
from simulation.simulate import simulate
st.set_page_config(page_title='PWR Core | Model 1',layout='wide',page_icon='⚛️')
st.title('PWR Core — Model 1')
st.caption('Interactive, reduced-order six-group neutronic–thermal educational benchmark. Not an AP1000-validated model or an operating tool.')
p=ReactorParameters()
with st.sidebar:
    st.header('Simulation settings')
    mode=st.radio('Control mode',['PI closed loop','Open-loop rod test'])
    duration=st.slider('Simulation length (s)',100,800,400,50)
    step_time=st.slider('Event time (s)',10,min(duration-10,250),50,10)
    st.subheader('Event input')
    if mode=='PI closed loop':
        delta_sp=st.slider('Outlet setpoint change (K)',-0.05,0.05,0.01,0.005,format='%.3f')
        rod_step=0.
    else:
        delta_sp=0.
        rod_step=st.slider('Commanded rod reactivity (pcm)',-1.0,1.0,0.5,0.1)
    delta_tin=st.slider('Inlet temperature disturbance (K)',-0.2,0.2,0.,0.02,format='%.2f')
    delta_flow=st.slider('Flow disturbance (%)',-2.,2.,0.,0.5,format='%.1f')
    st.subheader('PI parameters')
    kp=st.number_input('Kp (pcm/K)',min_value=0.,max_value=100.,value=30.,step=1.,disabled=mode!='PI closed loop')
    ki=st.number_input('Ki (pcm/K·s)',min_value=0.,max_value=10.,value=2.,step=0.1,disabled=mode!='PI closed loop')
    #go_button=st.button('Run simulation',type='primary',use_container_width=True)
    st.caption(
    'Every input change automatically recalculates the simulation. '
    'All nonzero event inputs activate together at the selected event time.'
)

    st.caption('Rod actuator: 2 s effective reactivity lag. Sensor: 1 s lag. No saturation/rod travel limit.')
scenario=Scenario(duration=float(duration),step_time=float(step_time),setpoint_step_k=float(delta_sp),
    inlet_step_k=float(delta_tin),flow_step_percent=float(delta_flow),rod_step_pcm=float(rod_step),
    closed_loop=mode=='PI closed loop')
control=ControlParameters(kp=float(kp),ki=float(ki))
# Sliders trigger a rerun; the button is an explicit scenario action as well.
try:
    result=simulate(scenario,p,control)
except Exception as exc:
    st.error(f'Simulation failed: {exc}')
    st.stop()
df=result.frame
m=result.metrics
cols = st.columns(4)

if scenario.closed_loop:
    cols[0].metric(
        'Peak tracking error',
        f"{m['peak_absolute_error_k']:.5f} K"
    )
    cols[1].metric(
        'Integrated absolute error',
        f"{m['iae_k_s']:.4f} K·s"
    )
    cols[2].metric(
        'Final tracking error',
        f"{m['final_error_k']:+.6f} K"
    )
else:
    cols[0].metric(
        'Peak temperature deviation',
        f"{m['peak_measured_deviation_k']:.5f} K"
    )
    cols[1].metric(
        'Integrated absolute deviation',
        f"{m['integrated_measured_deviation_k_s']:.4f} K·s"
    )
    cols[2].metric(
        'Final temperature deviation',
        f"{m['final_measured_deviation_k']:+.6f} K"
    )

cols[3].metric(
    'Final reactor power',
    f"{m['final_power_pu']:.6f} p.u."
)

def graph(names, title, ylabel):
    fig=go.Figure()
    for col,label in names:
        fig.add_trace(go.Scatter(x=df.time_s,y=df[col],mode='lines',name=label))
    fig.add_vline(x=scenario.step_time,line_dash='dot',line_color='gray')
    fig.update_layout(title=title,xaxis_title='Time (s)',yaxis_title=ylabel,
        legend=dict(orientation='h',y=-0.28),margin=dict(t=50,b=80),height=390)
    return fig

st.plotly_chart(graph([('setpoint_k','Setpoint'),('tmeas_k','Measured outlet'),('tout_k','True outlet')],
    'Outlet temperature','Temperature (K)'),use_container_width=True)
a,b=st.columns(2)
with a:
    st.plotly_chart(graph([('pn','Normalized power')],'Reactor power','Power (p.u.)'),use_container_width=True)
    st.plotly_chart(graph([('tf_k','Fuel'),('tc1_k','Lower coolant'),('tout_k','Upper coolant / outlet')],
        'Internal temperatures','Temperature (K)'),use_container_width=True)
with b:
    st.plotly_chart(graph([('rod_command_pcm','Commanded'),('rod_applied_pcm','Applied')],
        'Rod-reactivity actuator','Reactivity (pcm)'),use_container_width=True)
    st.plotly_chart(graph([('rod_applied_pcm','Applied rod'),('thermal_feedback_pcm','Thermal feedback'),
                           ('total_reactivity_pcm','Total')],
        'Reactivity breakdown','Reactivity (pcm)'),use_container_width=True)
st.download_button('Export simulation CSV',data=df.to_csv(index=False).encode(),
                   file_name='model_1_simulation.csv',mime='text/csv')
with st.expander('Model scope and limitations'):
    st.markdown('''- Six-group normalized point kinetics with one fuel and two well-mixed coolant states.
- Coefficients follow a *provisional educational benchmark*. The reference paper’s delayed neutron table contains a suspect fourth decay constant; not independently reconciled.
- PI gains are illustrative; rod drive is represented by an **assumed first-order reactivity lag**, not actual rod travel or rod worth.
- Pressure and steam-generator dynamics, actuator bounds, interlocks, rod speed, sensor noise and validated safety constraints are **not included**.
- Do **not** use for reactor operations, safety decisions, or AP1000 performance predictions. This is for educational and explorative purposes only.''')
