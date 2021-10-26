// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file contains public interface for target indepenent code generator.
 * @file generator.hpp
 */
#pragma once

#include <transformations_visibility.hpp>
#include "snippets_isa.hpp"
#include "emitter.hpp"

namespace ngraph {
namespace snippets {

TRANSFORMATIONS_API auto getRegisters(std::shared_ptr<ngraph::Node>& n) -> ngraph::snippets::RegInfo;

/**
 * @interface TargetMachine
 * @brief Base class Target machine representation. Target derives from this class to provide generator information about supported emittors
 * @ingroup snippets
 */
class TRANSFORMATIONS_API TargetMachine {
public:
    /**
     * @brief checks if target is natively supported
     * @return true, if supported
     */
    virtual bool is_supported() const = 0;

    /**
     * @brief finalizes code generation
     * @return generated kernel binary
     */
    virtual code get_snippet() const = 0;

    /**
     * @brief gets number of lanes supported by target's vector ISA
     * @return number of lanes
     */
    virtual size_t get_lanes() const = 0;

    /**
     * @brief called by generator to all the emittor for a target machine
     * @return a map by node's type info with callbacks to create an instance of emmitter for corresponding operation type
     */
    std::function<std::shared_ptr<Emitter>(std::shared_ptr<ngraph::Node>)> get(const ngraph::DiscreteTypeInfo type) const {
        auto jitter = jitters.find(type);
        if (jitter == jitters.end()) {
            throw ngraph_error(std::string("Target code emitter is not available for ") + type.name + " operation.");
        }
        return jitter->second;
    }

    /**
     * @brief checks if emitter for a specific operation is supported
     * @return true, if supported
     */
    bool has(const ngraph::DiscreteTypeInfo type) const {
        return jitters.find(type) != jitters.end();
    }

protected:
    // Fall back to old compare funcion, since it guarantees transitivity
    struct custom_less {
        bool operator()(const DiscreteTypeInfo &a, const DiscreteTypeInfo &b) const {
            return a.version < b.version || (a.version == b.version && strcmp(a.name, b.name) < 0);
        }
    };
    std::map<const ngraph::DiscreteTypeInfo, std::function<std::shared_ptr<Emitter>(std::shared_ptr<ngraph::Node>)>, custom_less> jitters;
};

/**
 * @interface Schedule
 * @brief Return scheduling information and pointer to generated kernel code
 * @ingroup snippets
 */
class TRANSFORMATIONS_API Schedule {
public:
    /**
     * @brief Default constructor
     */
    Schedule() : work_size({}), is_flat(false), ptr(nullptr) {}
    /**
     * @brief Default to create schedule out of specific parameters
     * @param ws work size for kernel execution
     * @param f can this kernel be linearided to 1D range
     * @param p pointer to generated code
     */
    Schedule(const Shape& ws, bool f, code p) : work_size(ws), is_flat(f), ptr(p) {}
    /**
     * @brief Returns callable instanse of code pointer
     */
    template<typename K> K get_callable() const {
        return reinterpret_cast<K>(const_cast<unsigned char*>(ptr));
    }

    Shape work_size {};
    bool is_flat {false};
    code ptr {nullptr};
};

/**
 * @interface Generator
 * @brief Target independent code generator interface
 * @ingroup snippets
 */
class TRANSFORMATIONS_API Generator {
public:
    /**
     * @brief Default constructor
     */
    Generator(const std::shared_ptr<TargetMachine>& t) : target(t) {}
    /**
     * @brief Default destructor
     */
    virtual ~Generator() = default;
    /**
     * @brief virtual method any specific implementation should implement
     * @param f runction in canonical for for table-based code generation
     * @return pointer to generated code
     */
    code generate(std::shared_ptr<Function>& f) const;

protected:
    std::shared_ptr<TargetMachine> target;
};

} // namespace snippets
} // namespace ngraph