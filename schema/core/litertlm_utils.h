#ifndef THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_PRINT_H_
#define THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_PRINT_H_

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "schema/core/litertlm_header_schema_generated.h"

// Helpful macros.

#if !defined(RETURN_IF_ERROR)
#define RETURN_IF_ERROR(EXPR) \
  if (auto s = (EXPR); !s.ok()) return s
#endif  // !defined(RETURN_IF_ERROR)

namespace litert {

namespace litertlm {

namespace schema {

// Utility function to provide a string name for AnySectionDataType enum
// values.
std::string AnySectionDataTypeToString(AnySectionDataType value);

}  // end namespace schema
}  // end namespace litertlm
}  // end namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_PRINT_H_
