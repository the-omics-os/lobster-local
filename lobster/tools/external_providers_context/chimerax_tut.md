# ChimeraX Python Command Reference for LLMs

## Quick Reference: Most Used Commands

```python
# ESSENTIAL IMPORTS
from chimerax.std_commands import open, close, color, show, hide, view, save
from chimerax.core.commands import run

# FETCH STRUCTURES FROM WEB (Most Common)
run(session, "open 1abc")                    # Fetch PDB structure by ID
run(session, "open alphafold:P12345")        # Fetch AlphaFold by UniProt ID
run(session, "open emdb:1234")               # Fetch EM map from EMDB

# Or using open_data
models, status = session.open_command.open_data('1abc', format='pdb')
```

## 1. FILE OPERATIONS

### Open Files
```python
# Open structure/map files
from chimerax.open_command.manager import OpenManager
session.open_command.open_data(path, format=None, **kw)
# Returns: (models, status_message)
# Formats: pdb, cif, mmcif, mrc, drc, pdb, mol2, sdf

# Example:
models, status = session.open_command.open_data('protein.pdb')
models, status = session.open_command.open_data('map.mrc', format='mrc')
```

### Fetch PDB Structures from Web
```python
# Fetch directly from RCSB PDB using PDB ID
from chimerax.core.commands import run

# Method 1: Using run command (simplest)
run(session, "open 1abc")  # Fetches PDB ID 1abc
run(session, "open 7bv2")  # Fetches PDB ID 7bv2

# Method 2: Using open_data with special syntax
models, status = session.open_command.open_data('1abc', format='pdb')
models, status = session.open_command.open_data('7bv2', format='pdb')

# Fetch from PDB and immediately return models
def fetch_pdb(session, pdb_id):
    """Fetch structure from RCSB PDB by ID"""
    try:
        # ChimeraX automatically recognizes 4-character codes as PDB IDs
        models, status = session.open_command.open_data(pdb_id, format='pdb')
        if models:
            return models[0]  # Return first model
        else:
            print(f"Failed to fetch PDB {pdb_id}: {status}")
            return None
    except Exception as e:
        print(f"Error fetching PDB {pdb_id}: {e}")
        return None

# Example usage
protein = fetch_pdb(session, '1abc')
if protein:
    print(f"Loaded {protein.name} with {len(protein.atoms)} atoms")

# Fetch multiple structures
pdb_ids = ['1abc', '2xyz', '3def']
structures = []
for pdb_id in pdb_ids:
    model = fetch_pdb(session, pdb_id)
    if model:
        structures.append(model)

# Fetch with specific options
run(session, "open 1abc structureModel true")  # As structure model
run(session, "open 1abc ignoreCache true")     # Force re-download
```

### Fetch AlphaFold Structures
```python
# Fetch AlphaFold predictions using UniProt ID
run(session, "open alphafold:P12345")  # UniProt accession

# Or using full AlphaFold database URL
af_url = "https://alphafold.ebi.ac.uk/files/AF-P12345-F1-model_v4.cif"
models, status = session.open_command.open_data(af_url, format='mmcif')
```

### Fetch Density Maps from EMDB
```python
# Fetch electron microscopy maps from EMDB
run(session, "open emdb:1234")  # EMDB ID

# Example: Fetch both structure and map
run(session, "open 7bv2")        # Structure from PDB
run(session, "open emdb:12345")  # Associated EM map
```

### Save Files
```python
from chimerax.save_command.manager import SaveManager
session.save_command.save_data(path, format=None, **kw)

# Example:
session.save_command.save_data('output.pdb', format='pdb')
session.save_command.save_data('image.png', format='png')
```

### Close Models
```python
from chimerax.std_commands import close
close(session, models=None)  # None = close all
# Example:
close(session, models=[model1, model2])
```

## 2. VISUALIZATION CONTROLS

### Show/Hide
```python
from chimerax.std_commands import show, hide

# Show objects
show(session, objects=None, what=None, only=False)
# what: 'atoms', 'bonds', 'cartoons', 'ribbons', 'surfaces', 'models'
# Example:
show(session, atoms, what='atoms')
show(session, atoms, what='cartoons')

# Hide objects
hide(session, objects=None, what=None)
# Example:
hide(session, atoms, what='atoms')
```

### Coloring
```python
from chimerax.std_commands import color

color(session, objects, color=None, what=None, transparency=None)
# color: 'red', 'blue', 'green', Color object, or special:
#   'byatom', 'byelement', 'byhetero', 'bychain', 'bypolymer'
# what: 'atoms', 'cartoons', 'ribbons', 'surfaces', 'bonds', 'models'

# Examples:
from chimerax.core.colors import Color
red = Color([255, 0, 0, 255])  # RGBA
color(session, atoms, color=red, what='atoms')
color(session, atoms, color='bychain', what='cartoons')
color(session, atoms, color='byelement', transparency=50)
```

### Cartoon/Ribbon Display
```python
from chimerax.std_commands import cartoon, uncartoon

cartoon(session, atoms=None, smooth=None, suppress_backbone_display=True)
# smooth: 0-1 (0=through atoms, 1=ideal geometry, default=0.7 for strands)

uncartoon(session, atoms=None)  # Hide ribbons
```

### Display Style
```python
from chimerax.std_commands import style

style(session, objects=None, atom_style=None)
# atom_style: 'sphere', 'ball', 'stick'

# Example:
style(session, atoms, atom_style='stick')
style(session, atoms, atom_style='sphere')
```

## 3. SURFACES

### Create Surface
```python
from chimerax.std_commands import surface

surface(session, atoms=None, probe_radius=1.4, resolution=None,
        grid_spacing=None, color=None, transparency=None)
# resolution: if set, creates Gaussian surface instead of SES

# Example:
surface(session, atoms, probe_radius=1.4, color='blue', transparency=50)
```

### Close Surface
```python
from chimerax.std_commands import surface_close
surface_close(session, objects=None)
```

### Color Surface
```python
# Color by electrostatics
from chimerax.std_commands import color_electrostatic
color_electrostatic(session, surfaces, map, palette=None, range=None)

# Color by map sampling
from chimerax.std_commands import color_sample
color_sample(session, surfaces, map, palette=None, range=None, offset=0)

# Color by radial distance
from chimerax.std_commands import color_radial
color_radial(session, surfaces, center=None, palette=None, range=None)
```

## 4. CAMERA & VIEW

### View Objects
```python
from chimerax.std_commands import view

view(session, objects=None, frames=None, clip=True, cofr=True, pad=0.05)
# frames: interpolate over N frames
# clip: enable front/back clipping
# cofr: set center of rotation

# Example:
view(session, atoms, frames=30, clip=True)  # Smooth transition
```

### Camera Movement
```python
from chimerax.std_commands import turn, move, zoom

# Rotate
turn(session, axis, angle=90, frames=None, center=None)
# axis: Axis object or (x,y,z) vector

# Translate
move(session, axis, distance=None, frames=None)

# Zoom
zoom(session, factor=None, frames=None)
# factor: >1 zooms in, <1 zooms out
```

### Camera Settings
```python
from chimerax.std_commands import camera

camera(session, type=None, field_of_view=None)
# type: 'mono', 'stereo', 'sbs' (side-by-side), 'tb' (top-bottom)
# field_of_view: degrees (horizontal)
```

## 5. SELECTION

### Select Objects
```python
from chimerax.std_commands import select

select(session, objects=None, residues=False)
# objects=None selects everything
# residues=True extends to full residues

# Get current selection:
sel_atoms = session.selection.items('atoms')
sel_residues = session.selection.items('residues')
sel_models = session.selection.items('models')
```

## 6. MEASUREMENTS

### Distance Measurement
```python
from chimerax.std_commands import measure_length
measure_length(session, bonds)
```

### SASA (Solvent Accessible Surface Area)
```python
from chimerax.std_commands import measure_sasa

measure_sasa(session, atoms=None, probe_radius=1.4,
             sum=None, set_attribute=True)
# Sets atom.area and residue.area attributes
```

### Buried Area
```python
from chimerax.std_commands import measure_buriedarea

measure_buriedarea(session, atoms1, with_atoms2=None,
                   probe_radius=1.4, list_residues=False)
```

## 7. STRUCTURE OPERATIONS

### Align Structures
```python
from chimerax.std_commands import align

align(session, atoms, to_atoms=None, move=None,
      match_chain_ids=False, cutoff_distance=None)
# move: 'structures', 'atoms', 'residues', 'chains', True, False
# Returns: (matched_atoms, matched_to_atoms, rmsd, paired_rmsd, transform)

# Example:
matched, matched_to, rmsd, pair_rmsd, xform = align(
    session, mobile_atoms, to_atoms=fixed_atoms, move='structures'
)
```

### Fit in Map
```python
from chimerax.std_commands import fitmap

fitmap(session, atoms_or_map, in_map=None, resolution=None,
       metric='correlation', search=0, move_whole_molecules=True)
# metric: 'overlap', 'correlation', 'cam'
# search: N random starting positions
```

## 8. DENSITY MAPS (VOLUME)

### Volume Display
```python
from chimerax.std_commands import volume

volume(session, volumes=None, style=None, show=None, hide=None,
       level=None, color=None, transparency=None, step=None)
# style: 'surface', 'mesh', 'image'
# level: contour level (can be list of levels)

# Example:
volume(session, [map_model], style='surface', level=0.5, color='cyan')
```

### Volume Operations
```python
# Mask a map
from chimerax.std_commands import mask
mask(session, volumes, surfaces, pad=0, invert_mask=False)

# Resample map
from chimerax.std_commands import volume_resample
volume_resample(session, volumes, on_grid=None, spacing=None)

# Zone around atoms
from chimerax.std_commands import volume_zone
volume_zone(session, volumes, near_atoms, range=3)
```

### Create Map from Atoms
```python
from chimerax.std_commands import molmap

molmap(session, atoms, resolution, grid_spacing=None,
       display_threshold=0.95)
# Creates Gaussian density map from atomic structure
```

## 9. LIGHTING & RENDERING

### Lighting
```python
from chimerax.std_commands import lighting

lighting(session, preset=None, direction=None, intensity=None,
         shadows=None, depth_cue=None)
# preset: 'default', 'simple', 'full', 'soft', 'flat'

# Example:
lighting(session, preset='soft', shadows=True)
```

### Material Properties
```python
from chimerax.std_commands import material

material(session, preset=None, reflectivity=None,
         specular_reflectivity=None)
# preset: 'default', 'shiny', 'dull', 'chimera'
```

### Background & Effects
```python
from chimerax.std_commands import set, graphics

set(session, bg_color=None, silhouettes=None, selection_color=None)
# Example:
from chimerax.core.colors import Color
white = Color([255, 255, 255, 255])
set(session, bg_color=white, silhouettes=True)

graphics(session, background_color=None)
```

## 10. LABELS

### 3D Atom Labels
```python
from chimerax.label.label3d import label, label_delete

label(session, objects=None, text=None, color=None,
      bg_color=None, size=None, height=0.7)
# object_type: 'atoms', 'residues', 'models'
# height: in scene units
# attribute: attribute name to display

# Example:
label(session, atoms, object_type='atoms', attribute='name')
label(session, residues, text='My Label', height=1.0, color='yellow')

label_delete(session, objects=None)
```

### 2D Screen Labels
```python
from chimerax.label.label2d import label_create, label_delete

label_create(session, name, text='', color=None, size=24,
             xpos=0.5, ypos=0.5)
# xpos, ypos: 0-1 screen coordinates

label_delete(session, labels=None)
```

## 11. MOVIE RECORDING

### Record Movies
```python
from chimerax.movie.moviecmd import movie_record, movie_encode, movie_stop

# Start recording
movie_record(session, directory=None, format='ppm',
             size=None, supersample=1)

# Encode to video
movie_encode(session, output='movie.mp4', format='h264',
             quality='good', framerate=25)

# Stop recording
movie_stop(session)
```

## 12. UTILITY FUNCTIONS

### Run Command String
```python
from chimerax.core.commands import run

run(session, "color red", log=True)
run(session, "select #1/A", log=True)
# Executes ChimeraX command-line syntax
```

### Log Messages
```python
from chimerax.log.cmd import log

log(session, text="My message", html=False)
log(session, thumbnail=True, width=200, height=200)
```

### Coordinate Systems
```python
# Get model transform
model.scene_position  # Place object (4x4 matrix)

# Axis and Center objects for geometric operations
from chimerax.core.commands import Axis, Center
axis = Axis((0, 0, 1))  # Z-axis
center = Center((0, 0, 0))  # Origin
```

## 13. COMMON WORKFLOWS

### Fetch & Display Protein from PDB
```python
# Fetch from RCSB PDB
from chimerax.core.commands import run
run(session, "open 1abc")  # Automatically fetches from PDB

# Or using open_data
models, status = session.open_command.open_data('1abc', format='pdb')
protein = models[0]

# Display as cartoon
from chimerax.std_commands import cartoon, color, hide
atoms = protein.atoms
hide(session, atoms, what='atoms')
cartoon(session, atoms)
color(session, atoms, color='bychain', what='cartoons')

# Center view
from chimerax.std_commands import view
view(session, protein)
```

### Load & Display Local Protein File
```python
# Open local PDB file
models, status = session.open_command.open_data('protein.pdb')
protein = models[0]

# Display as cartoon
from chimerax.std_commands import cartoon, color, hide
atoms = protein.atoms
hide(session, atoms, what='atoms')
cartoon(session, atoms)
color(session, atoms, color='bychain', what='cartoons')

# Center view
from chimerax.std_commands import view
view(session, protein)
```

### Load Density Map & Fit Structure
```python
# Open map
map_models, _ = session.open_command.open_data('map.mrc')
density_map = map_models[0]

# Open structure
struct_models, _ = session.open_command.open_data('model.pdb')
structure = struct_models[0]

# Display map
from chimerax.std_commands import volume
volume(session, [density_map], style='surface', level=0.5, transparency=50)

# Fit structure in map
from chimerax.std_commands import fitmap
fitmap(session, structure.atoms, in_map=density_map, resolution=3.0)
```

### Create Publication Figure
```python
# Set white background
from chimerax.std_commands import set
from chimerax.core.colors import Color
white = Color([255, 255, 255, 255])
set(session, bg_color=white, silhouettes=True)

# Nice lighting
from chimerax.std_commands import lighting
lighting(session, preset='soft')

# Save high-res image
session.save_command.save_data('figure.png', width=3000, height=3000,
                               supersample=3)
```

### Measure Interface Area
```python
# Select chain A and B
chainA_atoms = structure.atoms.filter(structure.atoms.chain_ids == 'A')
chainB_atoms = structure.atoms.filter(structure.atoms.chain_ids == 'B')

# Compute buried area
from chimerax.std_commands import measure_buriedarea
measure_buriedarea(session, chainA_atoms, with_atoms2=chainB_atoms,
                   list_residues=True, color='red')
```

### Compare Two PDB Structures (Fetch & Align)
```python
from chimerax.core.commands import run
from chimerax.std_commands import align, color, cartoon

# Fetch two related structures from PDB
run(session, "open 1abc")  # Reference structure
run(session, "open 2xyz")  # Structure to align

# Get models (assumes they're the first two opened)
ref_model = session.models.list()[0]
mobile_model = session.models.list()[1]

# Align structures
matched, matched_to, rmsd, pair_rmsd, xform = align(
    session,
    mobile_model.atoms,
    to_atoms=ref_model.atoms,
    move='structures'
)

print(f"RMSD: {rmsd:.2f} Å")
print(f"Matched {len(matched)} atoms")

# Color differently to distinguish
color(session, ref_model.atoms, color='cyan', what='cartoons')
color(session, mobile_model.atoms, color='magenta', what='cartoons')

# Display as cartoons
cartoon(session, ref_model.atoms)
cartoon(session, mobile_model.atoms)
```

## 14. COMMON PATTERNS

### Atom/Residue Filtering
```python
# Get all atoms
all_atoms = model.atoms

# Filter by element
ca_atoms = all_atoms.filter(all_atoms.element_names == 'C')
ca_atoms = all_atoms.filter(all_atoms.names == 'CA')  # CA backbone

# Filter by chain
chain_a = all_atoms.filter(all_atoms.chain_ids == 'A')

# Filter by residue
protein_atoms = all_atoms.filter(all_atoms.residues.polymer_types != 'none')
water = all_atoms.filter(all_atoms.residues.names == 'HOH')

# Filter by attribute
polar_atoms = all_atoms.filter(all_atoms.bfactors > 50)
```

### Color Schemes
```python
# Predefined colors (strings)
'red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'white', 'black'
'orange', 'purple', 'pink', 'lime', 'navy', 'gray'

# Special coloring modes
'byatom', 'byelement', 'byhetero', 'bychain', 'bypolymer',
'byidentity', 'bynucleotide', 'bymodel'

# Create custom color
from chimerax.core.colors import Color
custom = Color([128, 64, 192, 255])  # RGBA 0-255
```

## 15. ERROR HANDLING

```python
try:
    models, status = session.open_command.open_data('file.pdb')
    if not models:
        print(f"No models loaded: {status}")
except Exception as e:
    print(f"Error loading file: {e}")

# Check if atoms exist before operations
if len(atoms) == 0:
    print("No atoms selected")
else:
    color(session, atoms, color='red')
```

## 16. KEY OBJECTS & ATTRIBUTES

### Session Object
```python
session.models                # Model manager
session.selection            # Selection manager
session.open_command         # File opening
session.save_command         # File saving
session.logger              # Logging
```

### Structure Model
```python
model.atoms                  # All atoms
model.residues              # All residues
model.chains                # All chains
model.bonds                 # All bonds
model.name                  # Model name
model.id                    # Model ID tuple
model.scene_position        # Transform matrix
```

### Atoms Collection
```python
atoms.coords                 # Nx3 array of coordinates
atoms.elements              # Element objects
atoms.element_names         # Element symbols
atoms.names                 # Atom names
atoms.residues              # Parent residues
atoms.chain_ids             # Chain identifiers
atoms.bfactors              # B-factors
atoms.occupancies           # Occupancies
atoms.colors                # RGBA colors
```

### Residues Collection
```python
residues.names              # Residue names
residues.numbers            # Residue numbers
residues.chain_ids          # Chain IDs
residues.polymer_types      # Polymer types
residues.atoms              # Atoms in residues
```

## NOTES FOR LLM USAGE

1. **Always import needed functions** before calling them
2. **session** is the main ChimeraX session object - always required as first argument
3. **Objects** can be atoms, bonds, models - check type with isinstance()
4. **Coordinates**: Use scene coordinates unless specified otherwise
5. **Colors**: Can be strings ('red'), Color objects, or special keywords ('bychain')
6. **None arguments**: Often means "apply to all" or "use default"
7. **Return values**: Many functions return results - capture when needed
8. **Error handling**: Wrap file operations and selections in try/except

## QUICK COMMAND CHAINING EXAMPLE

```python
# Complete visualization pipeline
from chimerax.std_commands import *

# Load
models, _ = session.open_command.open_data('protein.pdb')
protein = models[0]

# Style
hide(protein.atoms, what='atoms')
cartoon(session, protein.atoms)
color(session, protein.atoms, color='bychain', what='cartoons')

# Surface
surface(session, protein.atoms, transparency=70, color='gray')

# View
view(session, protein)

# Lighting
lighting(session, preset='soft')
set(session, silhouettes=True)

# Save
session.save_command.save_data('output.png', supersample=3)
```

This is a focused reference covering 90% of common ChimeraX visualization tasks for LLM-driven analysis.
