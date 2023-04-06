def setup_dataset_args(args):
    if args.design_name in ["adaptec1", "bigblue1"]:
        args.num_bin_x = args.num_bin_y = 512
        args.target_density = 1.0
    elif args.design_name in ["adaptec2", "adaptec3", "adaptec4", "bigblue2"]:
        args.num_bin_x = args.num_bin_y = 1024
        args.target_density = 1.0
    elif args.design_name in ["bigblue3", "bigblue4"]:
        args.num_bin_x = args.num_bin_y = 2048
        args.target_density = 1.0
    elif args.design_name in ["mgc_des_perf_1"]:
        args.num_bin_x = args.num_bin_y = 512
        args.target_density = 0.91
    elif args.design_name in ["mgc_fft_1"]:
        args.num_bin_x = args.num_bin_y = 512
        args.target_density = 0.835
    elif args.design_name in ["mgc_fft_2"]:
        args.num_bin_x = args.num_bin_y = 512
        args.target_density = 0.65
    elif args.design_name in ["mgc_fft_a"]:
        args.num_bin_x = args.num_bin_y = 512
        args.target_density = 0.5
    elif args.design_name in ["mgc_fft_b"]:
        args.num_bin_x = args.num_bin_y = 512
        args.target_density = 0.6
    elif args.design_name in ["mgc_matrix_mult_1"]:
        args.num_bin_x = args.num_bin_y = 512
        args.target_density = 0.802
    elif args.design_name in ["mgc_matrix_mult_2"]:
        args.num_bin_x = args.num_bin_y = 512
        args.target_density = 0.8
    elif args.design_name in ["mgc_matrix_mult_a"]:
        args.num_bin_x = args.num_bin_y = 512
        args.target_density = 0.6
    elif args.design_name in ["mgc_superblue12"]:
        args.num_bin_x, args.num_bin_y = 1024, 1024
        args.target_density = 0.65
    elif args.design_name in ["mgc_superblue14"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.56
    elif args.design_name in ["mgc_superblue19"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.53
    elif args.design_name in ["mgc_des_perf_a"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.429
    elif args.design_name in ["mgc_des_perf_b"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.497
    elif args.design_name in ["mgc_edit_dist_a"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.455
    elif args.design_name in ["mgc_matrix_mult_b"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.60
    elif args.design_name in ["mgc_matrix_mult_c"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.60
    elif args.design_name in ["mgc_pci_bridge32_a"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.384
    elif args.design_name in ["mgc_pci_bridge32_b"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.143
    elif args.design_name in ["mgc_superblue11_a"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.65
    elif args.design_name in ["mgc_superblue16_a"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.55
    return args