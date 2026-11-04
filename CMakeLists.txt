cmake_minimum_required(VERSION 3.16)


# Set project languages conditionally based on platform
if(APPLE)
    project(whillats VERSION 1.0.0 LANGUAGES CXX C OBJC OBJCXX)
else()
    project(whillats VERSION 1.0.0 LANGUAGES CXX C)
endif()

# Configure Metal backend for ggml on macOS (enable for non-Release, disable for Release)
if(APPLE)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

# Ensure Position Independent Code is used for all targets
# Necessary when linking static libs into a shared lib.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
include(ExternalProject) # Needed for ExternalProject_Add

# Configure outputs per-configuration using generator expressions
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_FRAMEWORK_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Frameworks/$<CONFIG>)

# TTS engine selection (mutually exclusive)
option(WHILLATS_PIPER "Enable Piper neural TTS engine (fast CPU, requires ONNX Runtime)" OFF)
option(WHILLATS_STYLETTS2 "Enable StyleTTS2 neural TTS engine (requires ONNX Runtime)" ON)
# Runs alongside the engine above: personalized (cloned) voice for the
# languages its bundle covers. Needs CMake >= 3.28 for pocket-tts-raven.
option(WHILLATS_POCKET_TTS "Enable Pocket TTS zero-shot voice cloning (Kyutai, via pocket-tts-raven)" OFF)

if(WHILLATS_PIPER)
    set(WHILLATS_STYLETTS2 OFF CACHE BOOL "" FORCE)
    message(STATUS "TTS engine: Piper (CPU-optimized)")
elseif(WHILLATS_STYLETTS2)
    message(STATUS "TTS engine: StyleTTS2")
else()
    message(STATUS "TTS engine: espeak-ng (fallback)")
endif()

# C++17 required when Piper or StyleTTS2 is enabled
if(WHILLATS_PIPER OR WHILLATS_STYLETTS2)
    set(CMAKE_CXX_STANDARD 17)
else()
    set(CMAKE_CXX_STANDARD 14)
endif()

add_compile_options(-Wno-incompatible-pointer-types-discards-qualifiers)
add_compile_options(-Wno-exit-time-destructors -Wno-global-constructors)

# Match the pre-C++11 ABI used by WebRTC's bundled clang so that symbols
# (e.g. std::string returns) link without mangling mismatches.
option(WHILLATS_OLD_ABI "Use pre-C++11 libstdc++ ABI for clang compatibility" OFF)
if(WHILLATS_OLD_ABI)
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()

# --- Dependency Declarations ---
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
file(MAKE_DIRECTORY ${THIRD_PARTY_DIR}) # Ensure third_party exists

# Make mtmd headers (clip.h, mtmd.h) available to all targets
include_directories(${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd)

FetchContent_Declare(
    whisper
    GIT_REPOSITORY https://github.com/ggerganov/whisper.cpp.git
    GIT_TAG        v1.8.3
    SOURCE_DIR     ${THIRD_PARTY_DIR}/whisper.cpp
    GIT_SHALLOW    TRUE              # Faster clone
)
FetchContent_Declare(
    llama
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp
    GIT_TAG        b11019            # Gemma 4 (text + vision/audio mtmd) support
    SOURCE_DIR ${THIRD_PARTY_DIR}/llama.cpp
    GIT_SHALLOW TRUE # Faster clone
)
FetchContent_Declare(
    espeak-ng-proj # Rename to avoid conflict with potential target name
    GIT_REPOSITORY https://github.com/espeak-ng/espeak-ng
    SOURCE_DIR ${THIRD_PARTY_DIR}/espeak-ng
    GIT_SHALLOW TRUE # Faster clone
)
FetchContent_Declare(
    pcaudiolib-proj # Rename to avoid conflict
    GIT_REPOSITORY https://github.com/espeak-ng/pcaudiolib
    SOURCE_DIR ${THIRD_PARTY_DIR}/pcaudiolib
    GIT_SHALLOW TRUE # Faster clone
)

# Declare Agora SDK (Only for Linux)
if(UNIX AND NOT APPLE)
    FetchContent_Declare(
        agora_sdk
        URL https://download.agora.io/rtsasdk/release/Agora-RTC-x86_64-linux-gnu-v4.4.30-20241024_101940-398537.tgz
        # Specify the subdirectory inside the archive that contains the SDK
        SOURCE_SUBDIR Agora-RTC-x86_64-linux-gnu-v4.4.30-20241024_101940-398537 
        # CMake should automatically handle extraction for .tgz
    )
endif()

# --- ONNX Runtime for Piper / StyleTTS2 / Pocket TTS ---
if(WHILLATS_PIPER OR WHILLATS_STYLETTS2 OR WHILLATS_POCKET_TTS)
    # pocket-tts-raven needs ORT >= 1.23 and must share this copy: two ORT
    # versions in one process collide on OrtGetApiBase.
    if(WHILLATS_POCKET_TTS)
        set(_whillats_ort_default "1.23.2")
    else()
        set(_whillats_ort_default "1.17.3")
    endif()
    set(ONNXRUNTIME_VERSION "${_whillats_ort_default}" CACHE STRING "ONNX Runtime version to download")

    # Check if user provided ONNXRUNTIME_DIR
    if(ONNXRUNTIME_DIR)
        set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include)
        set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_DIR}/lib)
        message(STATUS "Using user-provided ONNX Runtime at: ${ONNXRUNTIME_DIR}")
    else()
        # Auto-download pre-built ONNX Runtime
        if(APPLE)
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
                set(ONNXRT_PLATFORM "osx-arm64")
            else()
                set(ONNXRT_PLATFORM "osx-x86_64")
            endif()
        elseif(UNIX)
            if(GGML_CUDA)
                set(ONNXRT_PLATFORM "linux-x64-gpu")
            else()
                set(ONNXRT_PLATFORM "linux-x64")
            endif()
        endif()

        set(ONNXRUNTIME_URL
            "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-${ONNXRT_PLATFORM}-${ONNXRUNTIME_VERSION}.tgz")
        message(STATUS "Downloading ONNX Runtime ${ONNXRUNTIME_VERSION} from: ${ONNXRUNTIME_URL}")

        FetchContent_Declare(
            onnxruntime
            URL ${ONNXRUNTIME_URL}
        )
        FetchContent_GetProperties(onnxruntime)
        if(NOT onnxruntime_POPULATED)
            FetchContent_Populate(onnxruntime)
        endif()

        set(ONNXRUNTIME_INCLUDE_DIR ${onnxruntime_SOURCE_DIR}/include)
        set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/lib)
    endif()

    # Create imported target for ONNX Runtime
    add_library(onnxruntime_lib SHARED IMPORTED)
    if(APPLE)
        set_target_properties(onnxruntime_lib PROPERTIES
            IMPORTED_LOCATION ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.dylib
        )
    else()
        set_target_properties(onnxruntime_lib PROPERTIES
            IMPORTED_LOCATION ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
        )
    endif()
    set_target_properties(onnxruntime_lib PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_INCLUDE_DIR}
    )
    message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_LIB_DIR}")
endif()

# --- Populate and Build Dependencies ---

# llama.cpp (CMake)
# Always fetch llama.cpp for all platforms including iOS
set(LLAMA_STATIC ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
# Force static linking for iOS
if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(GGML_STATIC ON CACHE BOOL "Force static ggml for iOS" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Force static builds for iOS" FORCE)
endif()
if(UNIX AND NOT APPLE) # Linux specific
    if(GGML_CUDA)
        set(GGML_CUDA ON CACHE BOOL "Enable CUDA" FORCE)
    else()
        set(GGML_CUDA OFF CACHE BOOL "Disable CUDA" FORCE)
    endif()
else()
    set(GGML_CUDA OFF CACHE BOOL "Disable CUDA" FORCE)
endif()
FetchContent_MakeAvailable(llama)

# Fix C++17 template argument deduction issue in mtmd-audio.cpp
file(READ "${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/mtmd-audio.cpp" MTMD_AUDIO_CONTENT)
string(REPLACE 
    "std::vector data(filters.n_mel * filters.n_fft, 0.0f);" 
    "std::vector<float> data(filters.n_mel * filters.n_fft, 0.0f);" 
    MTMD_AUDIO_CONTENT "${MTMD_AUDIO_CONTENT}")
file(WRITE "${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/mtmd-audio.cpp" "${MTMD_AUDIO_CONTENT}")

# Fix ARC incompatibility issues in ggml-metal.m
# file(READ "${THIRD_PARTY_DIR}/llama.cpp/ggml/src/ggml-metal/ggml-metal.m" GGML_METAL_CONTENT)
# string(REPLACE "[ctx->mtl_lock release];" "// [ctx->mtl_lock release]; // ARC handles this" GGML_METAL_CONTENT "${GGML_METAL_CONTENT}")
# string(REPLACE "[ctx->mtl_library release];" "// [ctx->mtl_library release]; // ARC handles this" GGML_METAL_CONTENT "${GGML_METAL_CONTENT}")
# string(REPLACE "[ctx->mtl_device release];" "// [ctx->mtl_device release]; // ARC handles this" GGML_METAL_CONTENT "${GGML_METAL_CONTENT}")
# string(REPLACE "[desc release];" "// [desc release]; // ARC handles this" GGML_METAL_CONTENT "${GGML_METAL_CONTENT}")
# string(REPLACE "[buf release];" "// [buf release]; // ARC handles this" GGML_METAL_CONTENT "${GGML_METAL_CONTENT}")
# string(REPLACE "[heap->obj  release];" "// [heap->obj  release]; // ARC handles this" GGML_METAL_CONTENT "${GGML_METAL_CONTENT}")
# string(REPLACE "[heap->bufs release];" "// [heap->bufs release]; // ARC handles this" GGML_METAL_CONTENT "${GGML_METAL_CONTENT}")
# string(REPLACE "[ptr release];" "// [ptr release]; // ARC handles this" GGML_METAL_CONTENT "${GGML_METAL_CONTENT}")
# string(REPLACE "[mem_pool->heaps           release];" "// [mem_pool->heaps           release]; // ARC handles this" GGML_METAL_CONTENT "${GGML_METAL_CONTENT}")
# string(REPLACE "[mem_pool->heaps_to_remove release];" "// [mem_pool->heaps_to_remove release]; // ARC handles this" GGML_METAL_CONTENT "${GGML_METAL_CONTENT}")
# string(REPLACE "[metal_function release];" "// [metal_function release]; // ARC handles this" GGML_METAL_CONTENT "${GGML_METAL_CONTENT}")
# file(WRITE "${THIRD_PARTY_DIR}/llama.cpp/ggml/src/ggml-metal/ggml-metal.m" "${GGML_METAL_CONTENT}")
set_source_files_properties(
    ${THIRD_PARTY_DIR}/llama.cpp/ggml/src/ggml-metal/ggml-metal.m
    PROPERTIES COMPILE_FLAGS "-fno-objc-arc"
)

# Fix Clang bracket nesting level limit for Metal compilation on Apple platforms
if(APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fbracket-depth=2048")
    set(CMAKE_OBJC_FLAGS "${CMAKE_OBJC_FLAGS} -fbracket-depth=2048")
    # Also set it globally for all targets using C and Objective-C
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-fbracket-depth=2048>)
    add_compile_options($<$<COMPILE_LANGUAGE:OBJC>:-fbracket-depth=2048>)
    # Force set for Objective-C Metal files specifically
    set_source_files_properties("${THIRD_PARTY_DIR}/llama.cpp/ggml/src/ggml-metal/ggml-metal.m"
        PROPERTIES COMPILE_FLAGS "-fbracket-depth=2048")
    # Also set for the ggml-metal target
    set_target_properties(ggml-metal PROPERTIES COMPILE_FLAGS "-fbracket-depth=2048")
endif()

# whisper.cpp (CMake)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libs for whisper" FORCE)
set(WHISPER_STATIC ON CACHE BOOL "Build static libs for whisper" FORCE) # Explicitly static
if(UNIX AND NOT APPLE) # Linux specific
    if(GGML_CUDA)
        set(GGML_CUDA ON CACHE BOOL "Enable CUDA" FORCE)
    else()
        set(GGML_CUDA OFF CACHE BOOL "Disable CUDA" FORCE)
    endif()
else()
    set(GGML_CUDA OFF CACHE BOOL "Disable CUDA" FORCE)
endif()
# Disable Metal backend for whisper (whisper uses its own ggml copy)
set(GGML_METAL OFF CACHE BOOL "Disable ggml Metal backend" FORCE)
FetchContent_MakeAvailable(whisper)

# Build pcaudiolib and espeak-ng only on Linux (skip on macOS and iOS)
if(UNIX AND NOT APPLE)
    # pcaudiolib (Autotools)
    FetchContent_GetProperties(pcaudiolib-proj)
    if(NOT pcaudiolib-proj_POPULATED)
        FetchContent_MakeAvailable(pcaudiolib-proj)
        message(STATUS "pcaudiolib source downloaded to ${pcaudiolib-proj_SOURCE_DIR}")
    endif()
    
    set(PCAUDIOLIB_BUILD_DIR ${CMAKE_BINARY_DIR}/pcaudiolib-build)
    set(PCAUDIOLIB_INSTALL_DIR ${CMAKE_BINARY_DIR}/pcaudiolib-install)
    set(PCAUDIOLIB_LIB_PATH ${PCAUDIOLIB_BUILD_DIR}/src/.libs/libpcaudio.a)
    
    execute_process(
        COMMAND sh ./autogen.sh
        WORKING_DIRECTORY ${pcaudiolib-proj_SOURCE_DIR}
        RESULT_VARIABLE _autogen_res
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT _autogen_res EQUAL 0)
        message(FATAL_ERROR "autogen.sh for pcaudiolib failed: ${_autogen_res}")
    endif()

    ExternalProject_Add(pcaudiolib_external
        SOURCE_DIR ${pcaudiolib-proj_SOURCE_DIR}
        BINARY_DIR ${PCAUDIOLIB_BUILD_DIR}
        CONFIGURE_COMMAND bash -c "export CFLAGS='-fPIC -I${pcaudiolib-proj_SOURCE_DIR}/src/include' && export CXXFLAGS='-fPIC -I${pcaudiolib-proj_SOURCE_DIR}/src/include' && exec <SOURCE_DIR>/configure --with-pic --prefix='${PCAUDIOLIB_INSTALL_DIR}'"
        BUILD_COMMAND $(MAKE)
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS ${PCAUDIOLIB_LIB_PATH}
        LOG_CONFIGURE YES
        LOG_BUILD YES
    )

    add_library(pcaudio STATIC IMPORTED)
    set_target_properties(pcaudio PROPERTIES
        IMPORTED_LOCATION ${PCAUDIOLIB_LIB_PATH}
        INTERFACE_INCLUDE_DIRECTORIES ${pcaudiolib-proj_SOURCE_DIR}/src/include
    )
    add_dependencies(pcaudio pcaudiolib_external)

    # espeak-ng (CMake)
    set(CMAKE_C_COMPILER ${CMAKE_C_COMPILER} CACHE STRING "C compiler" FORCE)
    list(APPEND CMAKE_PROGRAM_PATH /usr/bin /bin)
    set(CMAKE_PROGRAM_PATH ${CMAKE_PROGRAM_PATH} CACHE STRING "Path for CMake to search for programs" FORCE)
    FetchContent_MakeAvailable(espeak-ng-proj)

    # Define paths for espeak-ng data
    set(ESPEAK_DATA_SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/espeak-ng-proj-build/espeak-ng-data)
    set(ESPEAK_DATA_DEST_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/espeak-ng-data)

    add_custom_command(
        OUTPUT ${ESPEAK_DATA_DEST_DIR}/phontab
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${ESPEAK_DATA_SOURCE_DIR} ${ESPEAK_DATA_DEST_DIR}
        DEPENDS espeak-ng
        COMMENT "Copying espeak-ng-data to ${ESPEAK_DATA_DEST_DIR}"
    )
    add_custom_target(copy_espeak_data ALL DEPENDS ${ESPEAK_DATA_DEST_DIR}/phontab)
endif()  # end if(UNIX AND NOT APPLE)

# Build espeak-ng on macOS when StyleTTS2 or Piper is enabled (needed for phonemization)
if(APPLE AND (WHILLATS_STYLETTS2 OR WHILLATS_PIPER) AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    # Use add_subdirectory directly to avoid cmake 4.x FetchContent binary-dir reuse issues.
    # The source is already present (either submodule or previously fetched).
    if(NOT TARGET espeak-ng)
        add_subdirectory(${THIRD_PARTY_DIR}/espeak-ng
                         ${CMAKE_BINARY_DIR}/_deps/espeak-ng-proj-build
                         EXCLUDE_FROM_ALL)
    endif()

    set(ESPEAK_DATA_SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/espeak-ng-proj-build/espeak-ng-data)
    set(ESPEAK_DATA_DEST_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/espeak-ng-data)

    add_custom_command(
        OUTPUT ${ESPEAK_DATA_DEST_DIR}/phontab
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${ESPEAK_DATA_SOURCE_DIR} ${ESPEAK_DATA_DEST_DIR}
        DEPENDS espeak-ng data
        COMMENT "Copying espeak-ng-data to ${ESPEAK_DATA_DEST_DIR}"
    )
    add_custom_target(copy_espeak_data_styletts2 ALL DEPENDS ${ESPEAK_DATA_DEST_DIR}/phontab)
    if(WHILLATS_STYLETTS2)
        message(STATUS "StyleTTS2: Building espeak-ng for macOS phonemization")
    else()
        message(STATUS "Piper: Building espeak-ng for macOS phonemization")
    endif()
endif()

# --- mtmd example shared library ---
find_package(Threads REQUIRED)

# Every mtmd source except the CLI entry points; the list grows with each
# llama.cpp release (mtmd-image, mtmd-helper-gen, models/*), so glob it.
file(GLOB MTMD_SOURCES
    ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/*.cpp
    ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd/models/*.cpp)
list(FILTER MTMD_SOURCES EXCLUDE REGEX "/(mtmd-cli|deprecation-warning|mtmd-debug)[^/]*\\.cpp$")
set(MTMD_VENDOR_LIBS vendor::hash vendor::miniaudio vendor::stb vendor::sheredom)
add_library(mtmd OBJECT ${MTMD_SOURCES})

target_link_libraries(mtmd PRIVATE ggml llama Threads::Threads ${MTMD_VENDOR_LIBS})

target_include_directories(mtmd PUBLIC
    ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd
    ${THIRD_PARTY_DIR}/llama.cpp
    ${THIRD_PARTY_DIR}/llama.cpp/include
    ${THIRD_PARTY_DIR}/llama.cpp/common
    ${THIRD_PARTY_DIR}/llama.cpp/ggml/include # Added for ggml.h
    ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd
    ${THIRD_PARTY_DIR}/llama.cpp/vendor
)

target_compile_features(mtmd PRIVATE cxx_std_17)

add_library(mtmd_shared STATIC $<TARGET_OBJECTS:mtmd>)
set_target_properties(mtmd_shared PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_compile_definitions(mtmd_shared PRIVATE LLAMA_BUILD)
target_link_libraries(mtmd_shared PRIVATE ggml llama Threads::Threads ${MTMD_VENDOR_LIBS})
install(TARGETS mtmd_shared ARCHIVE)


# --- Main Library Target ---
if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    # Add Objective-C ARC flag only on Apple platforms
    if(APPLE)
        add_compile_options(-fobjc-arc)
    endif()
    # Define source files conditionally based on platform
    # ================================================================
    # libwhillats.so — THIN client library (no AI models, no GPU)
    # Contains: API classes, IPC client stubs, TalkingFace, espeak fallback
    # ================================================================
    set(WHILLATS_CLIENT_SOURCES
        src/whillats.cc
        src/whillats_client.cc
        src/whillats_utils.cc
        src/talking_face.cc
        src/synthesis.cc
    )

    # On Apple, add platform-specific TTS (in-process, no server needed)
    if(APPLE)
        list(APPEND WHILLATS_CLIENT_SOURCES
            src/whillats_synth.mm
            src/whillats_ios.mm
        )
    endif()

    add_library(${PROJECT_NAME} SHARED ${WHILLATS_CLIENT_SOURCES})

    # ================================================================
    # whillats_server — FAT standalone binary (all AI: Whisper, Llama, TTS)
    # Can be built with or without CUDA independently
    # ================================================================
    set(WHILLATS_SERVER_SOURCES
        src/whillats_server.cc
        src/whisper_transcription.cc
        src/llama_device_base.cc
        src/whillats_utils.cc
    )

    # Add TTS engine to server
    if(WHILLATS_PIPER)
        list(APPEND WHILLATS_SERVER_SOURCES
            src/piper_tts.cc
            src/piper_subprocess.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/piper/src/piper.cpp)
    elseif(WHILLATS_STYLETTS2)
        list(APPEND WHILLATS_SERVER_SOURCES
            src/styletts2_tts.cc
            src/orpheus_tts.cc)
    endif()
    if(WHILLATS_POCKET_TTS)
        set(BUILD_SHARED_LIB ON CACHE BOOL "Build libpocket_tts for whillats_server" FORCE)
        add_subdirectory(third_party/pocket-tts-raven EXCLUDE_FROM_ALL)
        list(APPEND WHILLATS_SERVER_SOURCES src/pocket_voice_tts.cc)
        message(STATUS "TTS voice cloning: Pocket TTS")
    endif()
    # Always include espeak TTS as fallback in server
    if(UNIX AND NOT APPLE)
        list(APPEND WHILLATS_SERVER_SOURCES src/espeak_tts.cc)
    endif()

    add_executable(whillats_server ${WHILLATS_SERVER_SOURCES})

    # Bind libstdc++ symbols locally so they don't get interposed by the host
    # binary's libc++ (WebRTC is built with clang/libc++ and -fno-exceptions).
    if(UNIX AND NOT APPLE)
        target_link_options(${PROJECT_NAME} PRIVATE -Wl,-Bsymbolic)
    endif()

    # Prevent -isystem /usr/include from breaking #include_next in GCC stdlib headers
    # when CUDA toolkit headers live at /usr/include (propagated via CUDA::cudart)
    set_target_properties(${PROJECT_NAME} PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)

    # ================================================================
    # libwhillats.so — built with WebRTC's clang + libc++ to match directcall
    # ================================================================
    set(WEBRTC_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
    set(WEBRTC_CLANG "${WEBRTC_SRC}/third_party/llvm-build/Release+Asserts/bin/clang++")
    set(WEBRTC_SYSROOT "${WEBRTC_SRC}/build/linux/debian_bullseye_amd64-sysroot")
    set(WEBRTC_LIBCXX "${WEBRTC_SRC}/third_party/libc++/src/include")
    set(WEBRTC_LIBCXXABI "${WEBRTC_SRC}/third_party/libc++abi/src/include")

    # Thin library: -fno-exceptions to match directcall.
    # Reader thread uses C malloc/free only — no C++ allocations that could throw.
    if(NOT APPLE)
        target_compile_options(${PROJECT_NAME} PRIVATE -fno-exceptions -fPIC)
    endif()

    target_include_directories(${PROJECT_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/include
            ${THIRD_PARTY_DIR}/llama.cpp/vendor/stb
    )

    target_link_libraries(${PROJECT_NAME} PRIVATE pthread)

    # ================================================================
    # whillats_server configuration — fat binary, ALL AI libraries
    # ================================================================
    set_target_properties(whillats_server PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)

    # Ensure espeak data is copied before building the server
    if(UNIX AND NOT APPLE)
        add_dependencies(whillats_server copy_espeak_data)
    elseif(APPLE AND (WHILLATS_STYLETTS2 OR WHILLATS_PIPER))
        add_dependencies(whillats_server copy_espeak_data_styletts2)
    endif()

    target_include_directories(whillats_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${THIRD_PARTY_DIR}/whisper.cpp/include
        ${THIRD_PARTY_DIR}/whisper.cpp/ggml/include
        ${THIRD_PARTY_DIR}/llama.cpp/include
        ${THIRD_PARTY_DIR}/llama.cpp/ggml/include
        ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd
        ${THIRD_PARTY_DIR}/llama.cpp/vendor/stb
    )

    # Server TTS compile definitions and includes
    if(WHILLATS_PIPER)
        target_compile_definitions(whillats_server PRIVATE WHILLATS_PIPER)
        target_link_libraries(whillats_server PRIVATE onnxruntime_lib espeak-ng)
        target_include_directories(whillats_server PRIVATE
            ${ONNXRUNTIME_INCLUDE_DIR}
            ${THIRD_PARTY_DIR}/espeak-ng/src/include
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/piper/include
        )
    elseif(WHILLATS_STYLETTS2)
        target_compile_definitions(whillats_server PRIVATE WHILLATS_STYLETTS2)
        target_link_libraries(whillats_server PRIVATE onnxruntime_lib espeak-ng)
        target_include_directories(whillats_server PRIVATE
            ${ONNXRUNTIME_INCLUDE_DIR}
            ${THIRD_PARTY_DIR}/espeak-ng/src/include
        )
    endif()

    # Server links Whisper + Llama + TTS
    if(WHILLATS_POCKET_TTS)
        target_compile_definitions(whillats_server PRIVATE WHILLATS_POCKET_TTS)
        target_include_directories(whillats_server PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pocket-tts-raven/include)
        target_link_libraries(whillats_server PRIVATE pocket-tts-lib)
    endif()
    target_link_libraries(whillats_server
        PRIVATE
            whisper
            mtmd_shared
            llama
            ggml-base
            espeak-ng
            pthread
    )

    # Server: CUDA support (optional)
    if(GGML_CUDA)
        find_package(CUDAToolkit)
        if(CUDAToolkit_FOUND)
            target_link_libraries(whillats_server PRIVATE CUDA::cudart ggml-cuda)
            target_include_directories(whillats_server PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
            message(STATUS "whillats_server: CUDA enabled")
        endif()
    endif()

    # Server: platform link options
    if(APPLE)
        target_link_libraries(whillats_server PRIVATE
            "-framework Accelerate"
            "-framework Metal"
            "-framework MetalKit"
        )
    elseif(UNIX)
        target_link_libraries(whillats_server PRIVATE gomp stdc++)
    endif()

    # Library: platform link options (minimal)
    if(APPLE)
        target_link_libraries(${PROJECT_NAME} PRIVATE
            "-framework CoreAudio"
            "-framework AudioToolbox"
            "-framework AVFoundation"
        )
    endif()

    target_compile_options(whillats_server PRIVATE $<$<CONFIG:Debug>:-g -O0>)
    target_link_options(whillats_server PRIVATE $<$<CONFIG:Debug>:-g>)
    if(UNIX AND NOT APPLE)
        target_link_options(whillats_server PRIVATE
            -Wl,--allow-shlib-undefined
            -Wl,-rpath,$ORIGIN
            -Wl,-rpath,${CMAKE_BINARY_DIR}/lib/Debug
        )
        if(ONNXRUNTIME_LIB_DIR)
            target_link_options(whillats_server PRIVATE -Wl,-rpath,${ONNXRUNTIME_LIB_DIR})
        endif()
    endif()

endif()  # end if(NOT iOS)

# --- iOS Framework Build ---
if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    message(STATUS "Configuring for iOS framework build")
    # Force static builds for all dependencies
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Force static builds" FORCE)
    set(GGML_STATIC ON CACHE BOOL "Force static ggml build" FORCE)
    set(GGML_BUILD_SHARED_LIBS OFF CACHE BOOL "Force static ggml build" FORCE)
    set(LLAMA_STATIC ON CACHE BOOL "Force static llama build" FORCE)
    set(LLAMA_BUILD_SHARED_LIBS OFF CACHE BOOL "Force static llama build" FORCE)
    # Build whisper for iOS
    set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(WHISPER_STATIC ON CACHE BOOL "Force static whisper build" FORCE)
    # Skip building whisper here to avoid binary dir conflict
    # add_subdirectory(${THIRD_PARTY_DIR}/whisper.cpp ${CMAKE_BINARY_DIR}/_deps/whisper-build EXCLUDE_FROM_ALL)
    # Build llama for iOS
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Force static llama build" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    # Build llama inline for iOS
    # Skip building llama here to avoid target name conflict - FetchContent already handles it
    # add_subdirectory(${THIRD_PARTY_DIR}/llama.cpp ${CMAKE_BINARY_DIR}/_deps/llama-ios-build EXCLUDE_FROM_ALL)
    add_library(mtmd_obj OBJECT ${MTMD_SOURCES})
    target_link_libraries(mtmd_obj PRIVATE ggml llama ${MTMD_VENDOR_LIBS})
    target_include_directories(mtmd_obj PRIVATE
        "${THIRD_PARTY_DIR}/llama.cpp/include"
        "${THIRD_PARTY_DIR}/llama.cpp/common"
        "${THIRD_PARTY_DIR}/llama.cpp/ggml/include"
        "${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd"
        "${THIRD_PARTY_DIR}/llama.cpp/vendor"
    )
    set_target_properties(mtmd_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # Define whillats.framework
    add_library(${PROJECT_NAME} SHARED
        src/whisper_transcription.cc
        src/llama_device_base.cc
        src/whillats.cc
        src/whillats_ios.mm
        src/whillats_synth.mm
        src/whillats_utils.cc
    )
    set_target_properties(${PROJECT_NAME} PROPERTIES
        FRAMEWORK ON
        FRAMEWORK_VERSION A
        MACOSX_FRAMEWORK_IDENTIFIER ai.webrtcsays.whillats
        XCODE_ATTRIBUTE_DEFINES_MODULE YES
        XCODE_ATTRIBUTE_CLANG_ENABLE_MODULES YES
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE TTS_PLATFORMS=0 WEBRTC_IOS)
    # Link statically to whisper and llama for iOS to avoid dylib dependencies
    target_link_libraries(${PROJECT_NAME} PRIVATE whisper llama $<TARGET_OBJECTS:mtmd_obj> ${MTMD_VENDOR_LIBS}
        "-framework AVFoundation" "-framework Foundation" "-framework Accelerate"
        "-framework Metal" "-framework MetalKit" "-framework AudioToolbox"
        "-framework CoreAudio"
    )
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd # Added for clip.h
    )
endif()  # end if(iOS)

# Enable debug symbols for whillats main library in Debug builds
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Debug>:-O0>
)
target_link_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:-g>
)

# Add after project() declaration, before other settings
if(APPLE)
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
endif()

# Modify the target compile definitions
target_compile_definitions(${PROJECT_NAME} 
    PRIVATE 
        WHILLATS_BUILDING_DLL
    PUBLIC
        $<$<PLATFORM_ID:Darwin>:WHILLATS_DLL>
)

if(APPLE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE PLATFORM_DARWIN)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        "-framework Accelerate"
        "-framework Metal"
        "-framework MetalKit"
        "-framework CoreAudio"
        "-framework AudioToolbox"
        "-framework AVFoundation"
    )
    set(APPLE_RPATHS
        "-Wl,-rpath,@loader_path/third_party/whisper.cpp/build/src"
        "-Wl,-rpath,@loader_path/third_party/whisper.cpp/build/ggml/src"
        "-Wl,-rpath,@loader_path/third_party/llama.cpp/build/bin"
        "-Wl,-rpath,@loader_path/third_party/llama.cpp/build/ggml/src"
        "-Wl,-rpath,@loader_path/third_party/pcaudiolib/src"
    )
    if((WHILLATS_PIPER OR WHILLATS_STYLETTS2) AND ONNXRUNTIME_LIB_DIR)
        list(APPEND APPLE_RPATHS "-Wl,-rpath,${ONNXRUNTIME_LIB_DIR}")
    endif()
    target_link_options(${PROJECT_NAME} PRIVATE ${APPLE_RPATHS})
elseif(UNIX AND NOT APPLE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE PLATFORM_LINUX)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        gomp
        stdc++
        pthread
        dl
        m
    )
    target_link_directories(${PROJECT_NAME} PRIVATE
        /usr/lib/x86_64-linux-gnu
        ${CMAKE_BINARY_DIR}/lib/Debug
    )
    set(LINUX_LINK_OPTS
        -Wl,--allow-shlib-undefined
        -Wl,-rpath,$ORIGIN
        -Wl,-rpath,${CMAKE_BINARY_DIR}/lib/Debug
    )
    if((WHILLATS_PIPER OR WHILLATS_STYLETTS2) AND ONNXRUNTIME_LIB_DIR)
        list(APPEND LINUX_LINK_OPTS -Wl,-rpath,${ONNXRUNTIME_LIB_DIR})
    endif()
    target_link_options(${PROJECT_NAME} PRIVATE ${LINUX_LINK_OPTS})
endif()

# Add test executable
add_executable(test_whillats
    test/test_whillats.cc
    test/test_utils.cc
)
set_target_properties(test_whillats PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)

# Add compile definition for espeak data path to test executable as well
target_include_directories(test_whillats PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_whillats
    PRIVATE
        ${PROJECT_NAME}
        pthread
)

target_compile_options(test_whillats PRIVATE
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Debug>:-O0>
    -fno-exceptions
)
target_link_options(test_whillats PRIVATE
    $<$<CONFIG:Debug>:-g>
)

if(APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
# Add synthesis process executable
add_executable(synthesis
    src/synthesis.mm
)
# Link Objective-C++ frameworks for synthesis
target_link_libraries(synthesis
    PRIVATE
        "-framework AppKit"
        "-framework Foundation"
        "-framework AVFoundation"
)
set_target_properties(synthesis PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>
)

# Keep macOS RPATH settings
    # Ensure test executable loads libwhillats.dylib from ../lib and ../lib/<config>
    set_target_properties(test_whillats PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "@loader_path/../../lib;@loader_path/../../lib/$<CONFIG>"
    )
endif()

# test_whillats_server: verifies IPC protocol between client stubs and server
add_executable(test_whillats_server
    test/test_whillats_server.cc
    test/test_utils.cc
    src/whillats_client.cc
)
target_include_directories(test_whillats_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${THIRD_PARTY_DIR}/whisper.cpp/include
    ${THIRD_PARTY_DIR}/whisper.cpp/ggml/include
    ${THIRD_PARTY_DIR}/llama.cpp/include
    ${THIRD_PARTY_DIR}/llama.cpp/ggml/include
    ${THIRD_PARTY_DIR}/llama.cpp/tools/mtmd
    ${THIRD_PARTY_DIR}/llama.cpp/vendor/stb
)
target_link_libraries(test_whillats_server PRIVATE ${PROJECT_NAME} pthread)
target_compile_options(test_whillats_server PRIVATE $<$<CONFIG:Debug>:-g -O0>)
target_link_options(test_whillats_server PRIVATE $<$<CONFIG:Debug>:-g>)

# Final install block: framework-only on iOS, full on others
if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    install(TARGETS ${PROJECT_NAME}
        FRAMEWORK DESTINATION bin
    )
else()
    install(TARGETS ${PROJECT_NAME} test_whillats whillats_server test_whillats_server
        FRAMEWORK DESTINATION bin
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION bin
    )
    # Install synthesis target only on macOS where it exists
    if(APPLE)
        install(TARGETS synthesis
            RUNTIME DESTINATION lib
        )
    endif()
endif()



