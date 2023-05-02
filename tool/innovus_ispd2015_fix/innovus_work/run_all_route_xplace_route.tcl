set design_names [list mgc_des_perf_1 mgc_fft_1 mgc_fft_2 mgc_fft_a mgc_fft_b mgc_matrix_mult_1 mgc_matrix_mult_2 mgc_matrix_mult_a mgc_des_perf_a mgc_des_perf_b mgc_edit_dist_a mgc_matrix_mult_b mgc_matrix_mult_c mgc_pci_bridge32_a mgc_pci_bridge32_b mgc_superblue11_a mgc_superblue12 mgc_superblue14 mgc_superblue16_a mgc_superblue19]

set placers [list xplace_route ]

set_multi_cpu_usage -local_cpu 10

foreach placer $placers {
    foreach var $design_names {
        puts "=========== $placer/$var ==========="
        if [regexp {superblue} $var] {
            set_db design_process_node 28
        } else {
            set_db design_process_node 65
        }
        read_physical -lefs "../ispd2015_fix/$var/$var.lef"
        read_netlist -def "../ispd2015_fix_$placer/$var/$var.def"
        init_design
        route_design
        report_route -summary
        check_drc -limit 100000000
        write_def -routing "../ispd2015_fix_$placer/$var/$var.routed.def"
        reset_design
    }
}
puts "Finish all!"
