#include <cstddef>
#include <iostream>
#include <ostream>  // Added for ostream
#include <string>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_read.h"
#include "schema/core/litertlm_utils.h"

namespace litert {
namespace litertlm {
namespace schema {

// Helper function to print KeyValuePair data.
void PrintKeyValuePair(const KeyValuePair* kvp, std::ostream& output_stream) {
  if (!kvp) {
    output_stream << "  KeyValuePair: nullptr" << std::endl;
    return;
  }
  output_stream << "  Key: " << kvp->key()->str() << ", ";

  switch (kvp->value_type()) {
    case VData::VData_StringValue: {
      output_stream << "Value (String): "
                    << kvp->value_as_StringValue()->value()->c_str() << "\n";
      break;
    }
    case VData::VData_Int32: {
      output_stream << "Value (Int32): " << kvp->value_as_Int32()->value()
                    << "\n";
      break;
    }
    case VData::VData_Float32: {
      output_stream << "Value (Float): " << kvp->value_as_Float32()->value()
                    << "\n";
      break;
    }
    case VData::VData_Bool: {
      output_stream << "Value (Bool): " << kvp->value_as_Bool()->value()
                    << "\n";
      break;
    }
    case VData::VData_UInt64: {
      output_stream << "Value (Uint64): " << kvp->value_as_UInt64()->value()
                    << "\n";
      break;
    }
    default:
      output_stream << "Value (Unknown Type: \n";
      break;
  }
  output_stream << std::endl;
}

absl::Status ProcessLiteRTLMFile(const std::string& litertlm_file,
                                 std::ostream& output_stream) {
  LitertlmHeader header;
  int major_version, minor_version, patch_version;

  absl::Status status = ReadHeaderFromLiteRTLM(
      litertlm_file, &header, &major_version, &minor_version, &patch_version);

  if (!status.ok()) {
    return status;  // Return the error from ReadSectionsFromLiteRTLM
  }
  // Print system metadata.
  output_stream << "Version: " << major_version << "." << minor_version << "."
                << patch_version << "\n";

  if (header.metadata == nullptr) {
    ABSL_LOG(ERROR) << "header is null ";
    return absl::InvalidArgumentError("header is null");
  }

  auto system_metadata = header.metadata->system_metadata();

  if (system_metadata == nullptr) {
    ABSL_LOG(ERROR) << "sytem metadata is null ";
    return absl::InvalidArgumentError("system metadata is null");
  }

  // Get the entries vector
  auto entries = system_metadata->entries();

  // Check if the entries vector is not null and has elements
  if (entries && entries->size() > 0) {
    output_stream << "Iterating over SystemMetadata entries:" << std::endl;
    for (size_t i = 0; i < entries->size(); ++i) {
      const KeyValuePair* entry = entries->Get(i);
      PrintKeyValuePair(entry, output_stream);
    }
  } else {
    output_stream << "SystemMetadata has no entries." << std::endl;
  }

  // Print section information.
  auto section_metadata_obj = header.metadata->section_metadata();
  auto section_objects = section_metadata_obj->objects();
  output_stream << "Sections (" << section_objects->size() << "):\n";
  if (section_objects->size() == 0) {
    output_stream << "  <None>\n";
  } else {
    for (size_t i = 0; i < section_objects->size(); ++i) {
      auto sec_obj = section_objects->Get(i);
      output_stream << "  Section " << i << ":\n";
      output_stream << "    Items:\n";
      const auto& items = sec_obj->items();
      if (items && items->size() > 0) {
        for (size_t i = 0; i < items->size(); ++i) {
          const KeyValuePair* item = items->Get(i);
          PrintKeyValuePair(item, output_stream);
        }
      }

      output_stream << "    Begin Offset: " << sec_obj->begin_offset() << "\n";
      output_stream << "    End Offset:    " << sec_obj->end_offset() << "\n";
      output_stream << "    Data Type:      "
                    << AnySectionDataTypeToString(sec_obj->data_type()) << "\n";
      output_stream << "\n";
    }
  }

  return absl::OkStatus();
}

}  // namespace schema
}  // namespace litertlm
}  // namespace litert
