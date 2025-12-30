
import unittest
from unittest.mock import patch, MagicMock, call, ANY
import sys

# Add the current directory to path to ensure ttr_refactored can be found
sys.path.append('.')

# It's important to import the module *after* the path is set
import ttr_refactored

class TestTTRPipeline(unittest.TestCase):

    @patch('ttr_refactored.addon_utils')
    @patch('ttr_refactored.Euler')
    @patch('ttr_refactored.bpy')
    def test_full_pipeline(self, mock_bpy, mock_euler, mock_addon_utils):
        """
        Tests the entire ttr pipeline from start to finish
        by mocking the bpy environment.
        """
        # --- Arrange ---
        # 1. Mock command line arguments
        mock_argv = [
            'test_in.glb',
            'test_image.png',
            'test_out.glb',
            'test_baked.png'
        ]

        # 2. Mock the Blender environment and data structures
        mock_mesh = MagicMock(name='mock_mesh')
        mock_mesh.type = 'MESH'
        mock_mesh.data.vertices = [1, 2, 3] # for len() check
        mock_mesh.name = "TestMesh"

        mock_rig = MagicMock(name='mock_rig')
        mock_rig.name = 'rig'

        mock_metarig = MagicMock(name='mock_metarig')
        mock_metarig.name = 'metarig'

        mock_cam = MagicMock(name='mock_cam')
        mock_baked_img = MagicMock(name='mock_baked_img')
        
        # When rigify is checked, say it's not enabled the first time
        mock_addon_utils.check.return_value = ("rigify", False)

        # Configure the mock bpy object to return our mocks
        # The import operation should add the mesh to the scene's objects
        def import_side_effect(*args, **kwargs):
            mock_bpy.context.scene.objects = [mock_mesh]
        mock_bpy.ops.import_scene.gltf.side_effect = import_side_effect

        mock_bpy.context.view_layer.objects.active = mock_mesh
        mock_bpy.data.images.load.return_value = MagicMock(name='src_img')
        mock_bpy.data.images.new.return_value = mock_baked_img
        mock_bpy.data.cameras.new.return_value = MagicMock(name='cam_data')
        mock_bpy.data.objects.new.return_value = mock_cam
        
        # When a new armature is added, it becomes the active object
        def armature_add_side_effect(*args, **kwargs):
            mock_bpy.context.view_layer.objects.active = mock_metarig
        mock_bpy.ops.object.armature_human_metarig_add.side_effect = armature_add_side_effect

        # When rigify_generate is called, the 'rig' object should be findable
        def rig_get_side_effect(name):
            if name == "rig":
                return mock_rig
            return None
        mock_bpy.context.scene.objects.get.side_effect = rig_get_side_effect

        # --- Act ---
        ttr_refactored.main(mock_argv)

        # --- Assert ---
        # Verify the sequence of major operations
        
        # 1. Rigify check
        mock_addon_utils.check.assert_called_with("rigify")
        mock_addon_utils.enable.assert_called_with("rigify", default_set=True)

        # 2. Setup and Import
        mock_bpy.ops.wm.read_factory_settings.assert_called_with(use_empty=True)
        mock_bpy.ops.import_scene.gltf.assert_called_with(filepath='test_in.glb')
        mock_bpy.ops.object.transform_apply.assert_called_with(location=False, rotation=True, scale=True)

        # 3. Camera setup
        mock_bpy.data.cameras.new.assert_called_with("BakeCam")
        self.assertEqual(mock_bpy.context.scene.camera, mock_cam)
        
        # 4. Material and Baking
        mock_bpy.data.images.load.assert_called_with('test_image.png')
        mock_bpy.data.images.new.assert_called_with("BakedDiffuse", 2048, 2048)
        # Check that the bake node was made active before baking
        self.assertEqual(mock_bpy.context.scene.node_tree.nodes.active.image, mock_baked_img)
        mock_bpy.ops.object.bake.assert_called_with(
            type='DIFFUSE',
            pass_filter={'COLOR'},
            margin=4
        )
        mock_baked_img.save.assert_called()
        self.assertEqual(mock_baked_img.filepath_raw, 'test_baked.png')

        # 5. Rigging
        mock_bpy.ops.object.armature_human_metarig_add.assert_called()
        mock_bpy.ops.pose.rigify_generate.assert_called()
        
        # 6. Parenting
        # Check that the right objects were selected before parenting
        parent_call_args = mock_bpy.ops.object.parent_set.call_args
        self.assertEqual(parent_call_args, call(type='ARMATURE_AUTO'))
        
        # 7. Export
        mock_bpy.ops.export_scene.gltf.assert_called_with(
            filepath='test_out.glb',
            export_format='GLB',
            use_selection=True,
            export_apply=True
        )

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
