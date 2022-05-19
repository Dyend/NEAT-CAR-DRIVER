from mujoco_py import load_model_from_xml, MjSim, MjViewer, load_model_from_path
import math
import os

#model = load_model_from_xml(MODEL_XML)
xml_path = './models/autito.xml'
model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)
t = 0
while True:
    sim.step()
    viewer.render()
    sim.data.ctrl[1] = 1
    sim.data.ctrl[0] = -1
    sim.data.qpos[15] = 0.5 + math.cos(t*0.01)
    t+=1
    print(sim.data.ctrl)
    print(sim.data.qpos)
    print(sim.get_state())