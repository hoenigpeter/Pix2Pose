for i in {1..30}; do
    DISPLAY=:0 python3 tools/4_converter_working.py 0 cfg/cfg_bop2020_rgb_custom.json tless $i
done