import plotly
import numpy as np

# Create custom Viridis colormap with black for 0 values
custom_viridis = np.array(plotly.colors.sequential.Viridis)
custom_viridis[0] = '#000000'  # Set black for 0 values

custom_magma = np.array(plotly.colors.sequential.Magma)
custom_magma[0] = '#000000'  # Set black for 0 values

custom_hot = plotly.colors.sequential.Hot
custom_hot[0] = '#000000'  # Set black for 0 values

custom_plotly3 = np.array(plotly.colors.sequential.Inferno)
custom_plotly3[0] = '#000000'  # Set black for 0 values

custom_aggrnyl = np.array(plotly.colors.sequential.Aggrnyl)
custom_aggrnyl[0] = '#000000'  # Set black for 0 values

custom_magma = np.array(plotly.colors.sequential.Magma)
custom_magma[0] = '#000000'  # Set black for 0 values

custom_cividis = np.array(plotly.colors.sequential.Cividis)
custom_cividis[0] = '#000000'  # Set black for 0 values

custom_plasma = np.array(plotly.colors.sequential.Plasma)
custom_plasma[0] = '#000000'  # Set black for 0 values

custom_jet = np.array(plotly.colors.sequential.Jet)
custom_jet[0] = '#000000'  # Set black for 0 values

custom_greysr = plotly.colors.sequential.Greys_r
custom_greysr[0] = '#000000'  # Set black for 0 values