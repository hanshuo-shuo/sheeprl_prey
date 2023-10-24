from cellworld import *


class Paths_builder (JsonObject):
    def __init__(self):
        self.moves = Coordinates_list()
        self.steps = JsonList(list_type=int)

    def get_from_name(world_configuration_name: str, occlussions_name: str, path_type_name: str = "astar"):
        if not type(world_configuration_name) is str:
            raise "incorrect type for world_configuration_name"
        if not type(occlussions_name) is str:
            raise "incorrect type for occlussions_name"
        if not type(path_type_name) is str:
            raise "incorrect type for path_type_name"
        return Paths_builder.parse(json_dictionary=get_resource("paths", world_configuration_name, occlussions_name, path_type_name))


class myPaths:
    def __init__(self, builder: Paths_builder, world: World):
        self.moves = builder.moves
        self.steps = builder.steps
        self.world = world
        self.map = Cell_map(world.configuration.cell_coordinates)

    def _get_index(self, s_ind, d_ind):
        computed_index = s_ind * len(self.world.cells) + d_ind
        return computed_index

    def get_move(self, src_cell: Cell, dst_cell: Cell):
        move_index = self._get_index(src_cell.id, dst_cell.id)
        move = self.moves[move_index]
        return move


    def get_path(self, src_cell: Cell, dst_cell: Cell):
        rst = Cell_group()
        curr_cell = src_cell
        move = self.get_move(curr_cell, dst_cell)
        curr_coordinates = src_cell.coordinates
        rst.append(curr_cell)
        while move != Coordinates(0, 0) and curr_cell != dst_cell:
            curr_coordinates = curr_coordinates + move
            new_cell_id = self.map[curr_coordinates]
            if new_cell_id == -1:
                break
            curr_cell = self.world.cells[new_cell_id]
            move_index = self._get_index(curr_cell.id, dst_cell.id)
            move = self.moves[move_index]
            rst.append(curr_cell)
        return rst
