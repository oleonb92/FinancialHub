import os
import sys
from sqlalchemy import create_engine, MetaData
from eralchemy2 import render_er

# Asegura que Python pueda encontrar backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Conecta a la base de datos
engine = create_engine("sqlite:///data/financialhub.db")
metadata = MetaData()
metadata.reflect(engine)

# Genera el diagrama ER
graph = render_er(metadata, "financialhub_erd.pdf")
print("✅ ERD generado exitosamente: financialhub_erd.pdf")