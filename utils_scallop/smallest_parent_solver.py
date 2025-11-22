from .scallop_solver import ScallopSolver


class SmallestParentSolver(ScallopSolver):
    def __init__(self, network, index_to_network_class, index_to_output_class, args):
        ScallopSolver.__init__(
            self, network, index_to_network_class, index_to_output_class, args
        )
        self.index_to_input_predicates = {k: f"image{k+1}" for k in range(self.M)}

        self.scl_ctx.add_relation("image1", (str), input_mapping=self.input_mapping)
        self.scl_ctx.add_relation("image2", (str), input_mapping=self.input_mapping)

        self.scl_ctx.add_rule('parent("truck") = image1("truck") and image2("truck")')
        self.scl_ctx.add_rule(
            'parent("automobile") = image1("automobile") and image2("automobile")'
        )
        self.scl_ctx.add_rule(
            'parent("airplane") = image1("airplane") and image2("airplane")'
        )
        self.scl_ctx.add_rule('parent("ship") = image1("ship") and image2("ship")')
        self.scl_ctx.add_rule('parent("cat") = image1("cat") and image2("cat")')
        self.scl_ctx.add_rule('parent("dog") = image1("dog") and image2("dog")')
        self.scl_ctx.add_rule('parent("deer") = image1("deer") and image2("deer")')
        self.scl_ctx.add_rule('parent("frog") = image1("frog") and image2("frog")')
        self.scl_ctx.add_rule('parent("horse") = image1("horse") and image2("horse")')
        self.scl_ctx.add_rule('parent("bird") = image1("bird") and image2("bird")')

        self.scl_ctx.add_rule(
            'parent("land_transportation") = image1("automobile") and image2("truck") or image1("truck") and image2("automobile")'
        )
        self.scl_ctx.add_rule(
            'parent("other_transportation") = image1("ship") and image2("airplane") or image1("airplane") and image2("ship")'
        )
        self.scl_ctx.add_rule(
            'parent("home_land_animals") = image1("cat") and image2("dog") or image1("dog") and image2("cat")'
        )
        self.scl_ctx.add_rule(
            'parent("wild_land_animals") = image1("deer") and image2("horse") or image1("horse") and image2("deer")'
        )
        self.scl_ctx.add_rule(
            'parent("other_animals") = image1("bird") and image2("frog") or image1("frog") and image2("bird")'
        )
        self.scl_ctx.add_rule(
            'parent("land_animals") = image1("home_land_animals") and image2("wild_land_animals") or image1("wild_land_animals") and image2("home_land_animals")'
        )
        self.scl_ctx.add_rule(
            'parent("animals") = image1("land_animals") and image2("other_animals") or image1("other_animals") and image2("land_animals")'
        )
        self.scl_ctx.add_rule(
            'parent("transportation") = image1("land_transportation") and image2("other_transportation") or image1("other_transportation") and image2("land_transportation")'
        )
        self.scl_ctx.add_rule(
            'parent("entities") = image1("transportation") and image2("animals") or image1("animals") and image2("transportation")'
        )

        self.scl_ctx.add_rule(
            'image1("home_land_animals") = image1("cat") or image1("dog")'
        )
        self.scl_ctx.add_rule(
            'image1("wild_land_animals") = image1("deer") or image1("horse")'
        )
        self.scl_ctx.add_rule(
            'image1("other_animals") = image1("bird") or image1("frog")'
        )
        self.scl_ctx.add_rule(
            'image1("land_animals") = image1("home_land_animals") or image1("wild_land_animals")'
        )
        self.scl_ctx.add_rule(
            'image1("animals") = image1("land_animals") or image1("other_animals")'
        )
        self.scl_ctx.add_rule(
            'image1("land_transportation") = image1("automobile") or image1("truck")'
        )
        self.scl_ctx.add_rule(
            'image1("other_transportation") = image1("airplane") or image1("ship")'
        )
        self.scl_ctx.add_rule(
            'image1("transportation") = image1("land_transportation") or image1("other_transportation")'
        )

        self.scl_ctx.add_rule(
            'image2("home_land_animals") = image2("cat") or image2("dog")'
        )
        self.scl_ctx.add_rule(
            'image2("wild_land_animals") = image2("deer") or image2("horse")'
        )
        self.scl_ctx.add_rule(
            'image2("other_animals") = image2("bird") or image2("frog")'
        )
        self.scl_ctx.add_rule(
            'image2("land_animals") = image2("home_land_animals") or image2("wild_land_animals")'
        )
        self.scl_ctx.add_rule(
            'image2("animals") = image2("land_animals") or image2("other_animals")'
        )
        self.scl_ctx.add_rule(
            'image2("land_transportation") = image2("automobile") or image2("truck")'
        )
        self.scl_ctx.add_rule(
            'image2("other_transportation") = image2("airplane") or image2("ship")'
        )
        self.scl_ctx.add_rule(
            'image2("transportation") = image2("land_transportation") or image2("other_transportation")'
        )

        self.set_context_forward_function("parent")
