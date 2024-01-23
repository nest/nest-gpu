#ifndef PROPAGATEERROR_H
#define PROPAGATEERROR_H

#define BEGIN_ERR_PROP                     \
  checkNESTGPUInstance();                  \
  NESTGPU_instance->SetErrorFlag( false ); \
  NESTGPU_instance->SetErrorMessage( "" ); \
  NESTGPU_instance->SetErrorCode( 0 );     \
  try

#define END_ERR_PROP                                                   \
  catch ( ngpu_exception & e )                                         \
  {                                                                    \
    NESTGPU_instance->SetErrorFlag( true );                            \
    NESTGPU_instance->SetErrorMessage( e.what() );                     \
    NESTGPU_instance->SetErrorCode( 2 );                               \
  }                                                                    \
  catch ( std::bad_alloc& )                                            \
  {                                                                    \
    NESTGPU_instance->SetErrorFlag( true );                            \
    NESTGPU_instance->SetErrorMessage( "Memory allocation error." );   \
    NESTGPU_instance->SetErrorCode( 1 );                               \
  }                                                                    \
  catch ( ... )                                                        \
  {                                                                    \
    NESTGPU_instance->SetErrorFlag( true );                            \
    NESTGPU_instance->SetErrorMessage( "Error in NESTGPU function." ); \
    NESTGPU_instance->SetErrorCode( 255 );                             \
  }                                                                    \
  if ( NESTGPU_instance->OnException() == ON_EXCEPTION_EXIT )          \
  {                                                                    \
    std::cerr << NESTGPU_instance->GetErrorMessage();                  \
    exit( NESTGPU_instance->GetErrorCode() );                          \
  }

#endif
