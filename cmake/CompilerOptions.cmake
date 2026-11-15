# Apply warnings only to project-owned C++ and CUDA compilation units.
function(fmm_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
        )
    else()
        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
        )
    endif()
endfunction()
