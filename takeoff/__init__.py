"""Plan-set PDF -> takeoff JSON.

The model labels line primitives; a takeoff needs counts, sizes, marks and room
areas. This package is everything between the two, with no torch dependency so
it can be tested and run on layer-derived labels alone:

    scale      drawing scale per viewport, from scale strings and dimension text
    stitch     overlapping tile predictions -> one label and object per primitive
    openings   door/window position, width, type mark, schedule row
    rooms      closed spaces from walls + openings, named from room tags
    document   the JSON: bim-ai field names, millimetres, floor-local Y-up

Ported, not imported, from bim-ai (extraction/dimension_parser.py,
geometry/room_builder.py, extraction/opening_detector.py) so the two repos stay
independent.
"""
