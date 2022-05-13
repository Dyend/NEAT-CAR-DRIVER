from mujoco_py import load_model_from_xml, MjSim, MjViewer, load_model_from_path
import math
import os

#model = load_model_from_xml(MODEL_XML)
xml_path = './models/autito.xml'
model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)
while True:
    sim.step()
    viewer.render()