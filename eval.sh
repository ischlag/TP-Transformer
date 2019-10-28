python3 main.py \
--model_name=tp-transformer \
--module_name=numbers__place_value \
--n_layers=6 \
--hidden=512 \
--filter=2048 \
--n_heads=8 \
--load_model="pretrained/tp-transformer_1.7M.pt" \
--batch_size=1024 \
--eval_mode