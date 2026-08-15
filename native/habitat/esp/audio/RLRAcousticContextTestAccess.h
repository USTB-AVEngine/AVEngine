// Copyright (c) Meta Platforms, Inc. and its affiliates.
// Modifications Copyright (c) AVEngine contributors.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef ESP_AUDIO_RLRACOUSTICCONTEXTTESTACCESS_H_
#define ESP_AUDIO_RLRACOUSTICCONTEXTTESTACCESS_H_

#include <string>

namespace esp {
namespace audio {
namespace detail {

/** Internal parser seam for adversarial OBJ fixture tests. */
std::string parseRLRSceneOBJMaterialCoefficientSha1ForTesting(
    const std::string& bytes);

}  // namespace detail
}  // namespace audio
}  // namespace esp

#endif  // ESP_AUDIO_RLRACOUSTICCONTEXTTESTACCESS_H_
