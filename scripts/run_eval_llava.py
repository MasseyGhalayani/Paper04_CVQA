def main(args):

    img_paths, queries, new_queries, answers, new_answers, typies = load_query_file(args.query)

    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device
    )

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    # Initialize conversation template based on model type.
    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    # ===== Changed: Instead of writing rows immediately to CSV, we accumulate results in a list =====
    results = []

    for (img_path, query, new_query, answer, new_answer, query_type) in tzip(img_paths, queries, new_queries, answers, new_answers, typies):

        image = load_image(os.path.join("../Paper04_CVQA/C-VQA/C-VQA-Synthetic/C-VQA-Synthetic_images", img_path))
        image_tensor = process_images([image], image_processor, model.config)
        if isinstance(image_tensor, list):
            image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        # Process the first query
        inp = make_prompt(query)
        conv = conv_templates[args.conv_mode].copy()
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

        result = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        print(inp, ":", result)

        # Process the new query
        inp = make_prompt(new_query)
        conv = conv_templates[args.conv_mode].copy()
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

        new_result = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        print(inp, ":", new_result)

        # ===== Changed: Instead of writer_res.writerow, we append to our results list =====
        results.append({
            'img_path': img_path,
            'query': query,
            'answer': answer,
            'new query': new_query,
            'new answer': new_answer,
            'type': query_type,
            'response': result,
            'new_response': new_result
        })

    # ===== Changed: Write all results at once using pandas =====
    df = pd.DataFrame(results)
    df.to_csv(f"{args.type}_responses.csv", index=False, encoding='utf-8')
