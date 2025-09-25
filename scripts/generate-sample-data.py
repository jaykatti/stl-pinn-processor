#!/usr/bin/env python3
"""
Generate sample data for testing STL-PINN Processor
"""
import sys
import json
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def create_sample_stl_files():
    """Create sample STL files for testing"""
    try:
        import open3d as o3d
        
        data_dir = Path("data/samples")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        print("📦 Generating sample STL files...")
        
        # Create a cube
        cube = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
        cube.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(data_dir / "cube.stl"), cube)
        print("  ✅ Created cube.stl")
        
        # Create a sphere
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=20)
        sphere.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(data_dir / "sphere.stl"), sphere)
        print("  ✅ Created sphere.stl")
        
        # Create a cylinder
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.3, height=1.0, resolution=20)
        cylinder.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(data_dir / "cylinder.stl"), cylinder)
        print("  ✅ Created cylinder.stl")
        
        # Create a more complex mesh (torus)
        torus = o3d.geometry.TriangleMesh.create_torus(torus_radius=0.5, tube_radius=0.2)
        torus.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(data_dir / "torus.stl"), torus)
        print("  ✅ Created torus.stl")
        
    except ImportError:
        print("❌ Open3D not installed. Run: pip install open3d")
        return False
    except Exception as e:
        print(f"❌ Error creating STL files: {e}")
        return False
    
    return True

def create_material_definitions():
    """Create material property definitions"""
    materials_dir = Path("data/materials")
    materials_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔧 Creating material definitions...")
    
    materials = {
        "steel": {
            "name": "Steel (Structural)",
            "density": 7850.0,  # kg/m³
            "youngs_modulus": 200e9,  # Pa
            "poisson_ratio": 0.3,
            "thermal_conductivity": 50.0,  # W/m·K
            "specific_heat": 500.0,  # J/kg·K
            "thermal_expansion": 12e-6,  # 1/K
            "yield_strength": 250e6,  # Pa
            "ultimate_strength": 400e6,  # Pa
            "description": "Structural steel for general engineering applications"
        },
        "aluminum": {
            "name": "Aluminum Alloy (6061-T6)",
            "density": 2700.0,
            "youngs_modulus": 70e9,
            "poisson_ratio": 0.33,
            "thermal_conductivity": 167.0,
            "specific_heat": 897.0,
            "thermal_expansion": 23e-6,
            "yield_strength": 276e6,
            "ultimate_strength": 310e6,
            "description": "Aluminum alloy for lightweight structures"
        },
        "aluminum": {
            "name": "Aluminum Alloy (6061-T6)",
            "density": 2700.0,
            "youngs_modulus": 70e9,
            "poisson_ratio": 0.33,
            "thermal_conductivity": 167.0,
            "specific_heat": 897.0,
            "thermal_expansion": 23e-6,
            "yield_strength": 276e6,
            "ultimate_strength": 310e6,
            "description": "Aluminum alloy for lightweight structures"
        },
        "titanium": {
            "name": "Titanium (Ti-6Al-4V)",
            "density": 4430.0,
            "youngs_modulus": 114e9,
            "poisson_ratio": 0.34,
            "thermal_conductivity": 6.7,
            "specific_heat": 563.0,
            "thermal_expansion": 8.6e-6,
            "yield_strength": 880e6,
            "ultimate_strength": 950e6,
            "description": "Titanium alloy for aerospace applications"
        },
        "copper": {
            "name": "Copper (Pure)",
            "density": 8960.0,
            "youngs_modulus": 110e9,
            "poisson_ratio": 0.34,
            "thermal_conductivity": 401.0,
            "specific_heat": 385.0,
            "thermal_expansion": 17e-6,
            "yield_strength": 70e6,
            "ultimate_strength": 220e6,
            "description": "Pure copper for electrical and thermal applications"
        },
        "plastic_abs": {
            "name": "ABS Plastic",
            "density": 1050.0,
            "youngs_modulus": 2.3e9,
            "poisson_ratio": 0.35,
            "thermal_conductivity": 0.25,
            "specific_heat": 1386.0,
            "thermal_expansion": 90e-6,
            "yield_strength": 40e6,
            "ultimate_strength": 45e6,
            "description": "ABS plastic for 3D printing and prototyping"
        }
    }
    
    for material_name, properties in materials.items():
        file_path = materials_dir / f"{material_name}.json"
        with open(file_path, 'w') as f:
            json.dump(properties, f, indent=2)
        print(f"  ✅ Created {material_name}.json")
    
    print("  ✅ All material definitions created")

def main():
    """Main function to generate all sample data"""
    print("🎯 Generating Sample Data for STL-PINN Processor")
    print("=" * 50)
    
    success = True
    
    # Generate STL files
    if not create_sample_stl_files():
        success = False
    
    # Generate material definitions
    create_material_definitions()
    
    # Generate PINN templates
    #create_pinn_templates()
    
    if success:
        print("\n✅ All sample data generated successfully!")
        print("\n📁 Generated files:")
        print("  - data/samples/*.stl - Sample mesh files")
        print("  - data/materials/*.json - Material property definitions")
        print("  - data/templates/*.py - PINN template files")
        print("\n🚀 Ready for testing and development!")
    else:
        print("\n⚠️ Some files could not be generated. Check dependencies.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())