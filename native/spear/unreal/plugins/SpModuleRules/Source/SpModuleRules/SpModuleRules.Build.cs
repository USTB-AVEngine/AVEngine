//
// Copyright (c) 2025 The SPEAR Development Team. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
// Copyright (c) 2022 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//

using System;                          // Console, Exception
using System.IO;                       // Path
using System.Runtime.CompilerServices; // CallerFilePath, CallerLineNumber, CallerMemberName
using UnrealBuildTool;                 // ModuleRules, ReadOnlyTargetRules

public class SpModuleRules : ModuleRules
{
    public SpModuleRules(ReadOnlyTargetRules readOnlyTargetRules) : base(readOnlyTargetRules)
    {
        // Disable precompiled headers entirely because they somehow force full rebuilds in UE 5.5.
        // Additionally, we prefer to avoid precompiled headers for easier debugging of compile errors, and
        // stricter enforcement of include-what-you-use.
        PCHUsage = PCHUsageMode.NoPCHs;

        // Disable unity builds for easier debugging of compile errors, and stricter enforcement of
        // include-what-you-use.
        bUseUnity = false;

        // Turn off code optimization except in shipping builds for faster build times.
        OptimizeCode = CodeOptimization.InShippingBuildsOnly;

        // Our error handling code throws exceptions, our SP_ASSERT macro throws exceptions, yaml-cpp (used
        // by Config) throws exceptions, and boost::interprocess::mapped_region (used by SharedMemoryRegion)
        // throws exceptions. So we enable exceptions everywhere.
        bEnableExceptions = true;

        // Required for:
        //     ... > SpCore/Std.h    > boost/tokenizer.hpp > ... > boost/exception/exception.h
        //     ... > SpCore/Rpclib.h > rpc/msgpack.hpp     > ... > rpc/msgpack/predef/other/endian.h
        UndefinedIdentifierWarningLevel = WarningLevel.Warning;

        PublicDependencyModuleNames.AddRange(new string[] {
            "AssetRegistry", "Chaos", "Core", "CoreUObject", "Engine", "EngineSettings", "InputCore", "Json", "JsonUtilities", "LevelSequence",
            "NavigationSystem", "PhysicsCore", "RenderCore", "RHI", "Slate"});
        PrivateDependencyModuleNames.AddRange(new string[] {});

        // Only add library dependencies if we're in a derived SpModuleRules class. This avoids build issues
        // on Linux where the SpModuleRules dummy plugin is trying to link against Boost but isn't set up
        // correctly, which can happen in UE 5.7.
        if (GetType().Name != "SpModuleRules" && GetType().Name != "SpModuleRulesEditor") {
            AddExternalSdkDependencies(readOnlyTargetRules);
        }
    }

    private void AddExternalSdkDependencies(ReadOnlyTargetRules readOnlyTargetRules)
    {
        // These roots are intentionally explicit. Do not search a SPEAR checkout, a sibling repository,
        // or an ambient package prefix: a user installs these ordinary C++ SDKs outside AVEngine.
        string boostRoot = RequireSdkRoot("AVENGINE_SPEAR_BOOST_ROOT", "include/boost/predef.h");
        PublicIncludePaths.Add(Path.Combine(boostRoot, "include"));

        string rpclibRoot = RequireSdkRoot("AVENGINE_SPEAR_RPCLIB_ROOT", "include/rpc/client.h");
        PublicIncludePaths.Add(Path.Combine(rpclibRoot, "include"));
        PublicAdditionalLibraries.Add(RequireStaticLibrary(
            "AVENGINE_SPEAR_RPCLIB_ROOT",
            rpclibRoot,
            readOnlyTargetRules.Platform,
            "lib/rpc.lib",
            "lib/librpc.a",
            "lib/librpc.a"));

        string yamlCppRoot = RequireSdkRoot("AVENGINE_SPEAR_YAML_CPP_ROOT", "include/yaml-cpp/yaml.h");
        PublicIncludePaths.Add(Path.Combine(yamlCppRoot, "include"));
        if (readOnlyTargetRules.Platform == UnrealTargetPlatform.Win64) {
            PublicDefinitions.Add("YAML_CPP_STATIC_DEFINE");
        }
        PublicAdditionalLibraries.Add(RequireStaticLibrary(
            "AVENGINE_SPEAR_YAML_CPP_ROOT",
            yamlCppRoot,
            readOnlyTargetRules.Platform,
            "lib/yaml-cpp.lib",
            "lib/libyaml-cpp.a",
            "lib/libyaml-cpp.a"));
    }

    private static string RequireSdkRoot(string environmentVariable, string requiredRelativePath)
    {
        string value = Environment.GetEnvironmentVariable(environmentVariable);
        if (String.IsNullOrWhiteSpace(value)) {
            throw new Exception(
                environmentVariable + " is required. It must name an absolute external installed SDK prefix; "
                + "this build never searches a SPEAR checkout or dependency source tree.");
        }
        if (!Path.IsPathRooted(value)) {
            throw new Exception(
                environmentVariable + " must be an absolute external SDK prefix, got: " + value);
        }

        string root = Path.GetFullPath(value);
        if (!Directory.Exists(root)) {
            throw new Exception(environmentVariable + " is not a directory: " + root);
        }

        string requiredPath = Path.Combine(root, requiredRelativePath);
        if (!File.Exists(requiredPath)) {
            throw new Exception(
                environmentVariable + " lacks required installed SDK file: " + requiredPath);
        }
        return root;
    }

    private string RequireStaticLibrary(
        string environmentVariable,
        string root,
        UnrealTargetPlatform platform,
        string windowsRelativePath,
        string macRelativePath,
        string linuxRelativePath)
    {
        string relativePath;
        if (platform == UnrealTargetPlatform.Win64) {
            relativePath = windowsRelativePath;
        } else if (platform == UnrealTargetPlatform.Mac) {
            relativePath = macRelativePath;
        } else if (platform == UnrealTargetPlatform.Linux) {
            relativePath = linuxRelativePath;
        } else {
            throw new Exception(SP_LOG_GET_PREFIX() + "Unexpected target platform: " + platform);
        }

        string libraryPath = Path.Combine(root, relativePath);
        if (!File.Exists(libraryPath)) {
            throw new Exception(
                environmentVariable + " lacks required static library: " + libraryPath);
        }
        return libraryPath;
    }

    protected void SP_LOG(string message, [CallerFilePath] string filePath="", [CallerLineNumber] int lineNumber=0)
    {
        Console.WriteLine(GetPrefix(filePath, lineNumber) + message);
    }

    protected void SP_LOG_CURRENT_FUNCTION([CallerFilePath] string filePath="", [CallerLineNumber] int lineNumber=0, [CallerMemberName] string memberName="")
    {
        Console.WriteLine(GetPrefix(filePath, lineNumber) + GetCurrentFunctionExpanded(memberName));
    }

    protected string SP_LOG_GET_PREFIX([CallerFilePath] string filePath="", [CallerLineNumber] int lineNumber=0)
    {
        return GetPrefix(filePath, lineNumber);
    }

    private string GetPrefix(string filePath, int lineNumber)
    {
        return "[SPEAR | " + GetCurrentFileAbbreviated(filePath) + ":" + lineNumber.ToString("D4") + "] ";
    }

    private string GetCurrentFileAbbreviated(string filePath)
    {
        return Path.GetFileName(filePath);
    }

    private string GetCurrentFunctionExpanded(string memberName)
    {
        string sep = memberName.StartsWith(".") ? "" : ".";
        return this.GetType() + sep + memberName;
    }
}
